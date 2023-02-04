import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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

def headLoss(y, p):
    p = torch.sigmoid(p)
    y = y.gt(0.5).float()
    inter = (p * y).mean(dim=[1,2,3])
    union = (p + y).mean(dim=[1,2,3]) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return (1.0 - iou).mean()

class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.initialize()

    def initialize(self):
        weight_init(self.head)

    def forward(self, x, global_step=0.0, mask=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        p = self.head(f2)

        if self.training:
            loss = headLoss(uphw(mask, p.shape[2::]).gt(0.5).float(), p)
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("loss", {"tot_loss": loss.item()}, global_step=global_step)
        else:
            loss = torch.zeros_like(p).mean()

        return {
            "loss": loss,
            "pred": torch.sigmoid(uphw(p, size=x.shape[2::]))
        }