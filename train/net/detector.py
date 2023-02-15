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
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.sal = ContrastiveSaliency(512, 8, 1024)
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.initialize()
        self.crf = CRF()
        self.lwt = LocalWindowTripleLoss(alpha=10.0)

    def initialize(self):
        weight_init(self.conv)
        weight_init(self.head)

    def forward(self, x, global_step=0.0, mask=None, epoch=1000, sw=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        del f5, f4, f3, f2; torch.cuda.empty_cache()
        y = self.head(f1)
        del f1; torch.cuda.empty_cache()

        if self.training:
            sure_mask = (torch.abs(mask-0.5) > 0.49) * 1.0
            sure_bce = (F.binary_cross_entropy_with_logits(uphw(y, size=mask.shape[2::]), mask, reduction="none") * sure_mask).sum() / (sure_mask.sum() + 1e-6)
            lwt_loss = self.lwt(torch.sigmoid(y), minMaxNorm(x), margin=0.5)
            loss = sure_bce + lwt_loss
            sw.add_scalars("loss", {"sure_bce_loss": sure_bce.item(), "pred#": torch.sigmoid(y).mean().item()})

        return {
            "loss": loss if self.training else 0.0,
            "pred": torch.sigmoid(uphw(y, size=x.shape[2::]))
        }