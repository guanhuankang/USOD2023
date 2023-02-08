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

        self.initialize()
        self.crf = CRF()
        self.lwt = LocalWindowTripleLoss(kernel_size=11)

    def initialize(self):
        weight_init(self.conv)

    def lwtLoss(self, y, feat):
        loss_record = {}
        N = len(feat) // 2
        lwtloss = self.lwt(torch.sigmoid(y), feat, margin=0.5); loss_record.update({"lwt_loss": lwtloss.item()})
        consloss = F.l1_loss(torch.sigmoid(y[0:N]), torch.sigmoid(y[N::])); loss_record.update({"cons_loss": consloss.item()})
        uncerloss = 0.5 - torch.abs(torch.sigmoid(y) - 0.5).mean(); loss_record.update({"uncertain_loss": uncerloss.item()})
        loss = lwtloss + consloss + uncerloss
        return loss, loss_record

    def forward(self, x, global_step=0.0, **kwargs):
        f1, f2, f3, f4, f5 = self.backbone(x)
        attn, loss = self.sal(self.conv(f5))
        preds = self.decoder([f1, f2, f3, f4, f5])

        if self.training:
            size = preds[0].shape[2::]
            ep_step = 1.0 / kwargs["epoches"]
            alpha_bce = delayWarmUp(step=global_step, period=ep_step * 10, delay=ep_step * 10)
            alpha_other = delayWarmUp(step=global_step, period=ep_step * 20, delay=ep_step * 20)
            loss_dict = {"clloss": loss.item()}

            sal_cues = self.crf(uphw(minMaxNorm(x), size=size), minMaxNorm(uphw(attn.detach(), size=size)), iters=10).gt(0.5).float() ## stop gradient
            if alpha_bce>1e-3 and alpha_bce<0.999:
                bceloss = sum([F.binary_cross_entropy_with_logits(uphw(y, size=size), sal_cues) for y in preds])
                loss_dict.update({"bce_loss": bceloss.item()})
                loss += bceloss
            if alpha_other>1e-3:
                p1_loss, loss_record = self.lwtLoss(preds[0], minMaxNorm(x))
                p2_loss, _ = self.lwtLoss(preds[1], minMaxNorm(x))
                p3_loss, _ = self.lwtLoss(preds[2], minMaxNorm(x))
                p4_loss, _ = self.lwtLoss(preds[3], minMaxNorm(x))
                loss += p1_loss + p2_loss + p3_loss + p4_loss

            loss_dict.update({"tot_loss": loss.item()})
            if "sw" in kwargs:
                kwargs["sw"].add_scalars("train_loss", loss_dict, global_step=global_step)

        return {
            "loss": loss,
            "pred": torch.sigmoid(uphw(preds[0], size=x.shape[2::])),
            "attn": attn
        }