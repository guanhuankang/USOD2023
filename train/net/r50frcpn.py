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
        self.decoder = FrcPN()

        self.conv = nn.Sequential(nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.sal = ContrastiveSaliency(512, 8, 1024)
        weight_init(self.conv)

        self.crf = CRF()
        self.lwt = LocalWindowTripleLoss(kernel_size=11)

    def lwtLoss(self, p, img):
        N = len(p) // 2
        y = torch.sigmoid(p)
        lwtloss = self.lwt(y, img, margin=0.5)
        consloss = F.l1_loss(y[0:N], y[N::])
        uncerloss = 0.5 - torch.abs(y - 0.5).mean()
        return lwtloss + consloss + uncerloss

    def CELoss(self, p, y):
        y = uphw(y.detach(), size=p.shape[2::])
        return F.binary_cross_entropy_with_logits(p, y.gt(0.5).float())

    def crfCELoss(self, p, y, img):
        size = p.shape[2::]
        y = self.crf(uphw(img, size=size), minMaxNorm(uphw(y.detach(), size=size)), iters=10).gt(0.5).float()
        return F.binary_cross_entropy_with_logits(p, y), float(y.mean())

    def forward(self, x, epoch=1000, **kwargs):
        info = ""
        f1, f2, f3, f4, f5 = self.backbone(x)
        attn, cl_loss = self.sal(self.conv(f5))
        p0, p1, p2, p3, p4, p5 = self.decoder([f1, f2, f3, f4, f5])

        if self.training:
            sal = minMaxNorm(attn.detach())
            img = minMaxNorm(x)

            bce_loss_0, s0 = self.crfCELoss(p0, sal, img) if epoch<=1 else (0.0, 0.0)
            bce_loss_1, s1 = self.crfCELoss(p1, sal, img)
            bce_loss = bce_loss_0 + bce_loss_1

            lwt_loss = self.lwtLoss(p0, img) if epoch>1 else 0.0

            loss = cl_loss + bce_loss + lwt_loss

            info = str(torch.sigmoid(bce_loss_0).mean())
            loss_dict = {
                "cl": float(cl_loss),
                "bce0": float(bce_loss_0),
                "bce1": float(bce_loss_1),
                # "bce2": float(bce_loss_2),
                # "bce3": float(bce_loss_3),
                "bce": float(bce_loss),
                "lwt": float(lwt_loss),
                "tot": float(loss)
            }
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("loss", loss_dict, global_step=kwargs["global_step"])

        return {
            "loss": loss if self.training else 0.0,
            "pred": torch.sigmoid(uphw(p0, size=x.shape[2::])),
            "attn": attn,
            "sal": s1 if self.training else 0.0,
            "loss_dict": loss_dict if self.training else {},
            "info": str(info)
        }