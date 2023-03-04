#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.base.modules import weight_init, LayerNorm2D

KEY = "conv+LN"

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
        self.conv = nn.Sequential(nn.Conv2d(2048, d_model, 1), nn.BatchNorm2d(d_model), nn.ReLU())
        self.transform = nn.Sequential(nn.Conv2d(d_model, d_model, 1), LayerNorm2D(d_model), nn.ReLU())
        self.conv_value = nn.Sequential(nn.Conv2d(d_model, d_model, 1), nn.BatchNorm2d(d_model), nn.ReLU())
        self.multi_head = nn.MultiheadAttention(d_model, n_head)
        self.g = PositionwiseFeedForward(d_model, d_ff)
        self.initialize()

    def initialize(self):
        weight_init(self.g)
        weight_init(self.conv)
        weight_init(self.transform)
        weight_init(self.conv_value)

    def forward(self, x, tau=0.1):
        # m = torch.softmax(self.qs(x).flatten(-2,-1), dim=-1).permute(2,0,1) ## hw,b,1
        batch, d_model, h, w = x.shape
        x = self.conv(x)

        mem = torch.flatten(self.transform(x), -2, -1).permute(2, 0, 1)  ## hw,b,d
        value = self.conv_value(x)

        q = torch.mean(mem, dim=0, keepdim=True) ## 1,b,d
        _, attn = self.multi_head(q, mem, mem) ## 1,b,d; b,1,hw
        attn = attn.reshape(batch, -1, h, w)
        out = torch.sum(attn * value, dim=[-1,-2]).unsqueeze(0) ## 1,b,d

        if self.training:
            y = F.normalize(self.g(out), p=2, dim=-1)  ## nq,batch,d_model
            cos_sim = torch.matmul(y, y.transpose(-1,-2))  ## nq,batch,batch
            cos_sim = cos_sim - 1e9 * torch.eye(batch, device=cos_sim.device)
            similarity = torch.softmax(cos_sim / tau, dim=-1)  ## nq, batch, batch
            prob = torch.diagonal(similarity, batch // 2, dim1=-2, dim2=-1)  ## nq, batch//2
            cl_loss = -torch.log(prob + 1e-6).mean()

            attn_sim_mse = nn.L1Loss()(attn[0:batch//2], attn[batch//2::]).mean()
            loss = attn_sim_mse + cl_loss
            return attn, loss
        else:
            return attn, torch.zeros_like(attn).sum()
