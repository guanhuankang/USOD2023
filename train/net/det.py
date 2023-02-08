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

def IOU(pred, target):
    inter = target * pred
    union = target + pred - target * pred
    iou_loss = 1 - torch.sum(inter, dim=(1, 2, 3)) / (torch.sum(union, dim=(1, 2, 3)) + 1e-7)
    return iou_loss.mean()

class Detector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN()

    def forward(self, x, global_step=0.0, mask=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        preds = self.decoder([f1,f2,f3,f4,f5])

        if self.training:
            loss = sum([IOU(torch.sigmoid(uphw(p, mask.shape[2::])), mask.gt(0.5).float()) for p in preds])
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("loss", {"tot_loss": loss.item()}, global_step=global_step)
        else:
            loss = torch.zeros_like(f1).mean()

        return {
            "loss": loss,
            "pred": torch.sigmoid(uphw(preds[0], size=x.shape[2::]))
        }