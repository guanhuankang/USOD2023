import torch
import torch.nn as nn
import torch.nn.functional as F
from base.modules import weight_init, LayerNorm2D, CRF
from base.resnet50fpn import FPN
from torch.utils.tensorboard import SummaryWriter

KEY = "convLN"

def min2D(m):
    return torch.min(torch.min(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

def max2D(m):
    return torch.max(torch.max(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

def minMaxNorm(m, eps=1e-12):
    return (m - min2D(m)) / (max2D(m) - min2D(m) + eps)

def warmUp(delay, warmPeriod, step):
    w = (step - delay) / warmPeriod
    return max(min(w, 1.0), 0.0)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class QSelection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        r = 16
        ph, pw = 128, 128
        self.instanceNorm = nn.InstanceNorm2d(d_model)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(d_model, d_model//r, 1),
            nn.BatchNorm2d(d_model//r),
            nn.ReLU(),
            nn.Conv2d(d_model//r, d_model, 1)
        )
        self.prior = nn.Parameter(torch.ones((1,1,ph,pw), dtype=torch.float32))
        weight_init(self.mlp)

    def forward(self, x, tau_channel=1.0):
        sq = self.gap(x)
        ex = torch.softmax(self.mlp(sq) / tau_channel, dim=1)
        iN = self.instanceNorm(x)
        m = torch.sum(iN * ex, dim=1, keepdim=True) * F.interpolate(self.prior, size=x.shape[-2::], mode="bilinear")
        return torch.sigmoid(m)

class ContrastiveSaliency(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.qs = QSelection(d_model)
        self.transform = {
            "enc": nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ff),
            "convLN": nn.Sequential(nn.Conv2d(d_model, d_model, 1), LayerNorm2D(d_model), nn.ReLU()),
            "none": nn.Identity()
        }[KEY]
        self.multi_head = nn.MultiheadAttention(d_model, n_head)
        self.g = PositionwiseFeedForward(d_model, d_ff)
        weight_init(self.g)

    def forward(self, x, tau=0.1):
        batch, d_model, h, w = x.shape
        m = torch.softmax(self.qs(x).flatten(-2,-1), dim=-1).permute(2,0,1) ## hw,b,1
        if KEY=="enc":
            x = torch.flatten(x, -2, -1).permute(2, 0, 1) ## hw,b,d
            mem = self.transform(x)
        else:
            x = self.transform(x)
            mem = torch.flatten(x, -2, -1).permute(2, 0, 1)  ## hw,b,d
        q = torch.sum(m * mem, dim=0, keepdim=True) ## 1,b,d
        out, attn = self.multi_head(q, mem, mem) ## 1,b,d; b,1,hw
        attn = attn.reshape(batch, -1, h, w)

        if self.training:
            y = F.normalize(self.g(out), p=2, dim=-1)  ## nq,batch,d_model
            cos_sim = torch.matmul(y, y.transpose(-1,-2))  ## nq,batch,batch
            cos_sim = cos_sim - 1e9 * torch.eye(batch, device=cos_sim.device)
            similarity = torch.softmax(cos_sim / tau, dim=-1)  ## nq, batch, batch
            prob = torch.diagonal(similarity, batch // 2, dim1=-2, dim2=-1)  ## nq, batch//2
            cl_loss = -torch.log(prob + 1e-6).mean()

            attn_sim_mse = nn.L1Loss()(attn[0:batch//2], attn[batch//2::]).mean()
            loss = attn_sim_mse + cl_loss
            return attn, loss
        else:
            return attn, torch.zeros_like(attn).sum()

class StageOneS521(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        self.fpn = FPN(cfg)
        self.clsal = ContrastiveSaliency(d_model=d_model, n_head=cfg.n_head, d_ff=cfg.d_ff)
        self.headLow = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
            nn.Conv2d(d_model, 1, 1)
        )
        self.headMid = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU(),
            nn.Conv2d(d_model, 1, 1)
        )
        self.crf = CRF()
        self.sw = SummaryWriter(cfg.eventPath)

    def calcLoss(self, pred, img, ref):
        N = len(pred) // 2
        size = (128,128)
        pred = F.interpolate(pred, size=size, mode="bilinear")
        img = F.interpolate(img, size=size, mode="bilinear")
        ref = F.interpolate(ref, size=size, mode="bilinear")
        cons_mse = nn.L1Loss()(torch.sigmoid(pred[0:N]), torch.sigmoid(pred[N::]))
        bceloss = F.binary_cross_entropy_with_logits(pred, self.crf(img, ref).gt(0.5).float())
        return bceloss + cons_mse

    def forward(self, x, global_step=0.0, **kwargs):
        img = minMaxNorm(x)
        low, mid, _, _, high = self.fpn(x)
        attn, loss = self.clsal(high)
        mid = self.headMid(mid)
        low = self.headLow(low)

        w_mid = warmUp(0.2, 0.05, global_step)
        w_low = warmUp(0.4, 0.05, global_step)
        loss_dict = {"clloss": loss.item(), "w_mid": w_mid, "w_low": w_low}

        if self.training and w_mid>1e-9:
            mid_loss = self.calcLoss(mid, img, minMaxNorm(attn))
            loss += mid_loss * w_mid
            loss_dict["mid_loss"] = mid_loss.item()
        if self.training and w_low>1e-9:
            low_loss = self.calcLoss(low, img, torch.sigmoid(mid))
            loss += low_loss * w_low
            loss_dict["low_loss"] = low_loss.item()

        loss_dict["tot_loss"] = loss.item()
        self.sw.add_scalars("loss", loss_dict, global_step=global_step)

        return {
            "loss": loss,
            "pred": torch.sigmoid(low),
            "preds": [minMaxNorm(attn), torch.sigmoid(mid), torch.sigmoid(low)]
        }
