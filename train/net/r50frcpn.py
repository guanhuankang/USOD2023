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
        self.lwt = LocalWindowTripleLoss(kernel_size=21)

    def lwtLoss(self, y, img):
        N = len(y) // 2
        y = torch.sigmoid(y)
        img = minMaxNorm(img)
        lwtloss = self.lwt(y, img, margin=0.5)
        consloss = F.l1_loss(y[0:N], y[N::])
        uncerloss = 0.5 - torch.abs(y - 0.5).mean()
        return lwtloss + consloss + uncerloss

    def forward(self, x, global_step=0.0, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        attn, loss = self.sal(self.conv(f5))
        preds = self.decoder([f1.detach(), f2.detach(), f3.detach(), f4.detach(), f5.detach()])
        loss_dict = {"clloss": loss.item()}

        if self.training:
            size = x.shape[2::]
            final = preds[0]
            sal_cues = self.crf(uphw(minMaxNorm(x), size=size), minMaxNorm(uphw(attn.detach(), size=size)),
                                iters=10).gt(0.5).float()  ## stop gradient

            auxloss = sum([F.binary_cross_entropy_with_logits(uphw(y, size=size), sal_cues) for y in preds[1::]])
            bceloss = F.binary_cross_entropy_with_logits(uphw(final, size=size), sal_cues)
            loss_dict.update({"bce_loss": bceloss.item()})

            lwtloss = self.lwtLoss(final, img=x)
            loss_dict.update({"lwtLoss": lwtloss.item()})

            alpha = float(min(torch.abs(torch.sigmoid(final).mean() - sal_cues.mean()) / (sal_cues.mean() + 1e-6), 1.0)) ## 0.0~1.0
            loss += alpha * bceloss + (1.0-alpha) * lwtloss + auxloss

            loss_dict.update({"tot_loss": loss.item()})
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("train_loss", loss_dict, global_step=global_step)

        return {
            "loss": loss,
            "pred": torch.sigmoid(uphw(preds[0], size=x.shape[2::])),
            "attn": attn
        }