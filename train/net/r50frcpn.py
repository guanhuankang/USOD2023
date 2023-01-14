import torch
import torch.nn as nn
import torch.nn.functional as F

from net.base.resnet50 import ResNet
from net.base.frcpn import FrcPN
from net.contrastive_saliency import ContrastiveSaliency
from net.base.modules import weight_init, CRF, LocalWindowTripleLoss

def delayWarmUp(step, period, delay):
    return min(1.0, max(0.0, 1./period * step - delay/period))

def min2D(m):
    return torch.min(torch.min(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

def max2D(m):
    return torch.max(torch.max(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]

def minMaxNorm(m, eps=1e-12):
    return (m - min2D(m)) / (max2D(m) - min2D(m) + eps)

def uphw(x, size):
    return F.interpolate(x, size=size, mode="bilinear")

class R50FrcPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_hidden = 512
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        self.g = nn.Sequential(nn.Conv2d(2048, d_hidden, 1), nn.ReLU(), nn.Dropout(0.1), nn.Conv2d(d_hidden, d_hidden, 1))
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.initialize()
        self.crf = CRF()
        self.lwt = LocalWindowTripleLoss(alpha=10.0)

    def initialize(self):
        weight_init(self.g)
        weight_init(self.head)

    def forward(self, x, global_step=0.0, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        y = self.head(f1)

        if self.training:
            ep_step = 1.0 / kwargs["epoches"]
            w = [1.0, 1.0]
            N = len(x) // 2
            alpha_other = delayWarmUp(step=global_step, period=ep_step * 4, delay=ep_step * 6)

            z = torch.mean(F.interpolate(torch.sigmoid(y), size=f5.shape[2::], mode="bilinear") * f5, dim=[-1,-2], keepdim=True)
            z = F.normalize(self.g(z), p=2, dim=1).unsqueeze(0).squeeze(-2).squeeze(-1)  ## 1,B,C
            cos_sim = torch.matmul(z, z.transpose(-1, -2))  ## nq=1,batch,batch
            cos_sim = cos_sim - 1e9 * torch.eye(N+N, device=cos_sim.device)
            similarity = torch.softmax(cos_sim / 0.1, dim=-1)  ## tau=0.1
            prob = torch.diagonal(similarity, N, dim1=-2, dim2=-1)  ## nq, batch//2
            loss = -torch.log(prob + 1e-6).mean()
            loss_dict = {"clloss": loss.item()}

            if alpha_other>1e-3:
                lwtloss = self.lwt(torch.sigmoid(y), minMaxNorm(x), margin=0.5); loss_dict.update({"lwt_loss": lwtloss.item()})
                uncerloss = 0.5 - torch.abs(torch.sigmoid(y) - 0.5).mean(); loss_dict.update({"uncertain_loss": uncerloss.item()})
                loss += lwtloss * w[0] + uncerloss * w[1]

            loss_dict.update({"tot_loss": loss.item()})
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("train_loss", loss_dict, global_step=global_step)
        else:
            loss = 0.0

        return {
            "loss": loss,
            "pred": uphw(torch.sigmoid(y), size=x.shape[2::])
        }