#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.base.modules import weight_init, LayerNorm2D

KEY = "conv"

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class QSelection(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        r = 16
        ph, pw = 128, 128
        self.instanceNorm = nn.InstanceNorm2d(d_model)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(d_model, d_model//r, 1),
            nn.BatchNorm2d(d_model//r),
            nn.ReLU(),
            nn.Conv2d(d_model//r, d_model, 1)
        )
        self.prior = nn.Parameter(torch.ones((1,1,ph,pw), dtype=torch.float32))
        weight_init(self.mlp)

    def forward(self, x, tau_channel=1.0):
        sq = self.gap(x)
        ex = torch.softmax(self.mlp(sq) / tau_channel, dim=1)
        iN = self.instanceNorm(x)
        m = torch.sum(iN * ex, dim=1, keepdim=True) * F.interpolate(self.prior, size=x.shape[-2::], mode="bilinear")
        return torch.sigmoid(m)

class ContrastiveSaliency(nn.Module):
    def __init__(self, d_model, n_head, d_ff):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(d_model, d_model//2, 1), nn.BatchNorm2d(d_model//2), nn.ReLU(),
            nn.Conv2d(d_model//2, 1, 1), nn.Sigmoid()
        )
        self.g = PositionwiseFeedForward(d_model, d_ff)
        weight_init(self.g)
        weight_init(self.head)

    def forward(self, x, tau=0.1):
        batch, d_model, h, w = x.shape
        attn = self.head(x) ## b, 1, h, w
        out = torch.sum(attn * x, dim=[-1,-2]) / (torch.sum(attn, dim=[-1,-2])+1e-9) ## b,d
        out = out.unsqueeze(0) ## 1,b,d

        if self.training:
            y = F.normalize(self.g(out), p=2, dim=-1)  ## nq,batch,d_model
            cos_sim = torch.matmul(y, y.transpose(-1,-2))  ## nq,batch,batch
            cos_sim = cos_sim - 1e9 * torch.eye(batch, device=cos_sim.device)
            similarity = torch.softmax(cos_sim / tau, dim=-1)  ## nq, batch, batch
            prob = torch.diagonal(similarity, batch // 2, dim1=-2, dim2=-1)  ## nq, batch//2
            cl_loss = -torch.log(prob + 1e-6).mean()

            attn_sim_mse = nn.L1Loss()(attn[0:batch//2], attn[batch//2::]).mean()
            l1loss = nn.L1Loss(attn, 0.5)
            amoloss = torch.abs(attn.mean(), 0.30)

            loss = attn_sim_mse + cl_loss + l1loss + amoloss
            return attn, loss
        else:
            return attn, torch.zeros_like(attn).sum()
