import torch
import torch.nn as nn
import torch.nn.functional as F

from net.base.resnet50 import ResNet
from net.base.frcpn import FRC, FrcPN
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

def iouLoss(pred, mask):
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3)) - inter
    iou  = 1.0-(inter+1e-6)/(union+1e-6)
    return iou.mean()

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        d_hidden = 3
        self.frc_high = FRC(2048, 1024)
        self.frc_mid = FRC(1024, 512)
        self.frc_low = FRC(512, 256)
        self.frc_lower = FRC(256, 64)
        self.frc_inp = FRC(64, d_hidden)
        self.inp = nn.Sequential(nn.Conv2d(3, d_hidden, 1), nn.BatchNorm2d(d_hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Conv2d(d_hidden, d_hidden, 1), nn.BatchNorm2d(d_hidden), nn.ReLU(),
            nn.Conv2d(d_hidden, 1, 1)
        )
        self.initialize()

    def initialize(self):
        weight_init(self.inp)
        weight_init(self.head)
    def forward(self, inp, f1, f2, f3, f4, f5):
        o = self.frc_high(f5, f4)
        o = self.frc_mid(o, f3)
        o = self.frc_low(o, f2)
        o = self.frc_lower(o, f1)
        o = self.frc_inp(o, self.inp(inp))
        o = self.head(o)
        return o

class R50FrcPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64]) ## self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.crf = CRF()
        self.lwt = LocalWindowTripleLoss(alpha=10.0)

        weight_init(self.head)

    def forward(self, x, global_step=0.0, mask=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        y = self.head(uphw(f1, size=x.shape[2::]))

        if self.training:
            size = x.shape[2::]
            mask = F.interpolate(mask, size=size, mode="bilinear").gt(0.5).float()
            y = F.interpolate(y, size=size, mode="bilinear")

            ep_step = 1.0 / kwargs["epoches"]
            w = [1.0, 1.0]
            N = len(x) // 2

            iouloss = F.binary_cross_entropy_with_logits(y, mask)
            consloss = F.l1_loss(torch.sigmoid(y[0:N]), torch.sigmoid(y[N::]))
            loss = w[0] * iouloss + w[1] * consloss

            if "sw" in kwargs:
                kwargs["sw"].add_scalars("train_loss", {
                    "bceloss": iouloss.item(),
                    "consloss": consloss.item(),
                    "tot_loss": loss.item()
                }, global_step=global_step)
        else:
            loss = 0.0

        return {
            "loss": loss,
            "pred": torch.sigmoid(uphw(y, size=x.shape[2::]))
        }