import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from net.base.resnet50 import ResNet
from net.base.frcpn import FrcPN
from net.contrastive_saliency import ContrastiveSaliency
from net.base.modules import weight_init, CRF, LocalWindowTripleLoss
from net.base.CBAM import CBAM

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

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()


class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        self.head = nn.Sequential(
            nn.Conv2d(256*3, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.fc1 = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU()
        )
        self.initialize()

    def initialize(self):
        weight_init(self.head)
        weight_init(self.fc1)
        weight_init(self.fc2)
        weight_init(self.fc3)

    def forward(self, x, global_step=0.0, mask=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        p = self.head(torch.cat([self.fc1(f1),self.fc2(f2),self.fc3(uphw(f3,size=f1.shape[2::]))], dim=1))

        if self.training:
            loss = IOU(torch.sigmoid(uphw(p, mask.shape[2::])), mask.gt(0.5).float())
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("loss", {"tot_loss": loss.item()}, global_step=global_step)
        else:
            loss = torch.zeros_like(p).mean()

        return {
            "loss": loss,
            "pred": torch.sigmoid(uphw(p, size=x.shape[2::]))
        }