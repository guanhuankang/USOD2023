#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels
from pydensecrf.utils import unary_from_softmax


def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            try:
                m.initialize()
            except:
                pass


class LayerNorm2D(nn.Module):
    "Construct a layernorm module"

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model).view(1, d_model, 1, 1))
        self.b_2 = nn.Parameter(torch.zeros(d_model).view(1, d_model, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class LayerNorm1D(nn.Module):
    "Construct a layernorm module"

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(d_model))
        self.b_2 = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.initialize()

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

    def initialize(self):
        weight_init(self)


class ExciteChannelWise(nn.Module):
    def __init__(self, d_model, r):
        super().__init__()
        self.excite = nn.Sequential(
            nn.Conv2d(d_model, d_model // r, 1),
            nn.ReLU(),
            nn.Conv2d(d_model // r, d_model, 1),
            nn.Sigmoid()
        )
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        weight_init(self)

    def forward(self, x):
        sq = self.squeeze(x)
        ex = self.excite(sq)
        scale = x * ex
        return scale


class ExcitePixelWise(nn.Module):
    def __init__(self, prior_size=(20, 20)):
        super().__init__()
        self.exciteFactor = nn.Parameter(torch.ones(1, 1, prior_size[0], prior_size[1]))

    def forward(self, x):
        X = torch.sum(x, dim=1, keepdim=True)
        mean = torch.mean(X, dim=[-1, -2], keepdim=True)
        std = torch.std(X, dim=[-1, -2], keepdim=True)
        sq = (X - mean) / (std + 1e-6)
        ex = torch.sigmoid(sq * F.interpolate(self.exciteFactor, size=sq.shape[2::], mode="bilinear"))
        scale = x * ex
        return scale


class Excitation(nn.Module):
    def __init__(self, d_model, r_channel=16):
        super().__init__()
        self.exciteChannel = ExciteChannelWise(d_model=d_model, r=r_channel)
        self.excitePixel = ExcitePixelWise()

    def forward(self, x):
        return self.excitePixel(x) + self.exciteChannel(x)

class UnFold(nn.Module):
    def __init__(self, kernel_size, dilation=1, padding=0, stride=1):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=padding, stride=stride)
        kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.window_size = kernel_size[0]*kernel_size[1]
        self.h = lambda x: math.floor((x+2*padding-dilation*(kernel_size[0]-1)-1)/stride + 1)
        self.w = lambda x: math.floor((x+2*padding-dilation*(kernel_size[1]-1)-1)/stride + 1)
    def forward(self, x):
        b,c,h,w = x.shape
        h, w = self.h(h), self.w(w)
        return self.unfold(x).contiguous().reshape(b,c,self.window_size,h,w) ## b,c,window_size,h,w

class CRF(nn.Module):
    def __init__(self, sxy_g=3, compat_g=3, sxy_b=50, srgb_b=5, compat_b=10):
        super().__init__()
        self.sxy_g = sxy_g
        self.compat_g = compat_g
        self.sxy_b = sxy_b
        self.srgb_b = srgb_b
        self.compat_b = compat_b

    def refine(self, img, labels, iters, gt_prob=0.5, n_labels=2):
        h,w = img.shape[:2]
        d = dcrf.DenseCRF2D(w, h, 2)
        # unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
        unary = unary_from_softmax(np.stack([labels.reshape(-1),1.0-labels.reshape(-1)], axis=0))
        d.setUnaryEnergy(unary)
        d.addPairwiseGaussian(sxy=self.sxy_g, compat=self.compat_g)
        d.addPairwiseBilateral(sxy=self.sxy_b, srgb=self.srgb_b, rgbim=img, compat=self.compat_b)
        infer = np.array(d.inference(iters)).astype('float32')
        res = infer[0, :].reshape(h, w)
        return res

    def forward(self, img, pred, iters=5):
        assert img.shape[2::]==pred.shape[2::]
        img_np = (img.detach().cpu().permute(0,2,3,1).numpy()*255).astype(np.uint8) ## b,h,w,c
        pred_np = pred.detach().cpu().squeeze(1).numpy().astype(np.float32) ## b,h,w
        crf_results = []
        for i in range(len(img_np)):
            res = self.refine(np.ascontiguousarray(img_np[i]), np.ascontiguousarray(pred_np[i]), iters=iters)
            crf_results.append(torch.tensor(res, dtype=img.dtype, device=img.device).unsqueeze(0))
        return torch.stack(crf_results, dim=0) ## b,1,h,w

class SmoothConv(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, dilation=1, padding=1, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(inplane, outplane, kernel_size, dilation=dilation, padding=padding, stride=stride)
        self.initialize()

    def initialize(self):
        w = self.conv.weight.data * 0.0 + 1.0
        self.conv.weight = nn.Parameter(w)
        self.conv.bias = nn.Parameter(self.conv.bias.data * 0.0)
        self.conv.requires_grad_(False)

    def forward(self, x):
        return self.conv(x) / 9.0

class LabelSmooth(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, smooth_factor):
        x = x.gt(0.5).float() ## binarization
        epilon = x.sum() * smooth_factor / ( (1.0-x).sum() + 1e-6)
        x = x * (1.0 - smooth_factor) + (1.0 - x) * epilon
        return x

class LocalWindowTripleLoss(nn.Module):
    def __init__(self, d_in, d_out=256, alpha=0.01):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_in, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.ReLU(),
            nn.Conv2d(d_out, d_out, 1)
        )
        self.unfold = UnFold(11, dilation=1, padding=0, stride=5)
        self.alpha = alpha

    def forward(self, map, feat, margin=0.5):
        m = F.interpolate(map, size=feat.shape[2::], mode="bilinear").gt(0.5).float()
        feat = self.conv(feat) ## b,c,h,w
        mu = self.unfold(m) ## b,1,w_s,h,w
        fu = self.unfold(feat) ## b,c,w_s,h,w
        pos_cnt = torch.sum(mu, dim=2, keepdim=True)
        neg_cnt = torch.sum(1.0 - mu, dim=2, keepdim=True)
        valid = ((pos_cnt > 0.5) * (neg_cnt > 0.5)).float()

        pc = torch.sum(mu * fu, dim=2, keepdim=True) / (pos_cnt + 1e-6) ## b,c,1,h,w
        nc = torch.sum((1.0-mu) * fu, dim=2, keepdim=True) / (neg_cnt + 1e-6) ## b,c,1,h,w
        dist_pc = 1.0 - torch.exp(-self.alpha*torch.sum(torch.pow(pc - fu, 2.0), dim=1, keepdim=True)) ## b,1,w_s,h,w
        dist_nc = 1.0 - torch.exp(-self.alpha*torch.sum(torch.pow(nc - fu, 2.0), dim=1, keepdim=True)) ## b,1,w_s,h,w
        pos_dist = torch.where(mu>0.5, dist_pc, dist_nc) ## b,1,w_s,h,w
        neg_dist = torch.where(mu>0.5, dist_nc, dist_pc) ## b,1,w_s,h,w
        triple = torch.maximum(pos_dist - neg_dist + margin, torch.zeros_like(pos_dist)) ## b,1,w_s,h,w
        loss = torch.sum(triple * valid.detach(), dim=[3,4]) / (torch.sum(valid.detach(), dim=[3,4]) + 1e-6)
        return loss.mean()