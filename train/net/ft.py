import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

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

def edgeMap(x):
    edgs = []
    for m in list(x.detach().cpu().permute(0,2,3,1)):
        e = cv2.Canny( (m.detach().numpy() * 255).astype(np.uint8), 100, 200)
        e = torch.tensor(e.astype(float) / 255.0).gt(0.5).float()
        edgs.append(e.unsqueeze(0))
    return torch.stack(edgs, dim=0).to(x.device)

def headLoss(y, p):
    p = torch.sigmoid(p)
    y = y.gt(0.5).float()
    inter = (p * y).mean(dim=[1,2,3])
    union = (p + y).mean(dim=[1,2,3]) - inter
    iou = (inter + 1e-6) / (union + 1e-6)
    return (1.0 - iou).mean()

class EDN(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.conv5 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 5, padding=2), nn.BatchNorm2d(d_model), nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, padding=1), nn.BatchNorm2d(d_model), nn.ReLU()
        )
        self.conv = nn.Conv2d(d_model, 1, 1)
        self.initilize()

    def initilize(self):
        weight_init(self.conv)
        weight_init(self.conv3)
        weight_init(self.conv5)

    def forward(self, x):
        return self.conv(self.conv5(x) - self.conv3(x))

class FT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = ResNet(cfg.backboneWeight)
        self.decoder = FrcPN(dim_bin=[2048,1024,512,256,64])
        d_model = 256
        self.heads = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(dim, d_model, 1), nn.BatchNorm2d(d_model), nn.ReLU(),
                nn.Conv2d(d_model, 1, 1)
            )
            for dim in [2048,1024,512,256,64]
        )
        self.edn = EDN(64)

        self.initialize()

    def initialize(self):
        for x in self.heads:
            weight_init(x)

    def forward(self, x, global_step=0.0, mask=None, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        f5, f4, f3, f2, f1 = self.decoder([f5, f4, f3, f2, f1])
        p5,p4,p3,p2,p1 = [ head(f) for head,f in zip(self.heads, [f5, f4, f3, f2, f1]) ]
        edg = self.edn(f1)

        if self.training:
            loss_lst = [headLoss(uphw(mask, p.shape[2::]).gt(0.5).float(), p) for p in [p5,p4,p3,p2,p1]]
            loss = sum(loss_lst)
            edg_loss = F.binary_cross_entropy_with_logits(edg, edgeMap(uphw(mask, edg.shape[2::])))
            loss += edg_loss
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("loss", {"tot_loss": loss.item(), "loss_lst": loss_lst[-1].item(), "edg_loss": edg_loss.item()}, global_step=global_step)
        else:
            loss = torch.zeros_like(p5).mean()

        return {
            "loss": loss,
            "pred": uphw(torch.sigmoid(p1), size=x.shape[2::]),
            "edge": uphw(torch.sigmoid(edg), size=x.shape[2::])
        }