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

def iouLoss(s, t):
    i = (s * t).sum()
    u = (s + t).sum() - i
    return 1.0 - (i+1e-6) / (u+1e-6)

class FineTune(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        self.head = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.initialize()
        self.crf = CRF()

    def initialize(self):
        weight_init(self.head)

    def forward(self, x, global_step=0.0, mask=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        y = self.head(f1)

        if self.training:
            size = mask.shape[2::]
            w = [1.0, 1.0, 0.0]
            N = len(x) // 2
            loss_dict = {}

            bceloss = F.binary_cross_entropy_with_logits(uphw(y, size), mask.gt(0.5).float()); loss_dict.update({"bce_loss": bceloss.item()})
            consloss = F.l1_loss(torch.sigmoid(y[0:N]), torch.sigmoid(y[N::])); loss_dict.update({"cons_loss": consloss.item()})
            iouloss = iouLoss(torch.sigmoid(uphw(y, size=size)), mask); loss_dict.update({"iou_loss": iouloss.item()})
            loss = w[0] * bceloss + w[1] * consloss + w[2] * iouloss; loss_dict.update({"tot_loss": loss.item()})

            if "sw" in kwargs:
                kwargs["sw"].add_scalars("train_loss", loss_dict, global_step=global_step)
        else:
            loss = 0.0

        return {
            "loss": loss,
            "pred": uphw(torch.sigmoid(y), size=x.shape[2::])
        }