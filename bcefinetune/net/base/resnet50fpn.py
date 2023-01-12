#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from net.base.resnet50 import ResNet, weight_init

def upx2(m):
    b, d, h, w = m.shape
    return F.interpolate(m, size=(2 * h, 2 * w), mode="bilinear")

class FPN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_model = cfg.d_model
        getnorm = lambda d: nn.BatchNorm2d(d)
        self.backbone = ResNet(cfg.backboneWeight)
        self.conv5 = nn.Sequential(nn.Conv2d(2048, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1024, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(512, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d(64, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv4_2 = nn.Sequential(nn.Conv2d(d_model, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(d_model, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(d_model, d_model, 1), getnorm(d_model), nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(d_model, d_model, 1), getnorm(d_model), nn.ReLU())
        self.initialize()

    def initialize(self):
        weight_init(self.conv1)
        weight_init(self.conv2)
        weight_init(self.conv3)
        weight_init(self.conv4)
        weight_init(self.conv5)
        weight_init(self.conv1_2)
        weight_init(self.conv2_2)
        weight_init(self.conv3_2)
        weight_init(self.conv4_2)

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.backbone(x)
        out1, out2, out3, out4, out5 = self.conv1(out1), self.conv2(out2), self.conv3(out3), self.conv4(
            out4), self.conv5(out5)
        out4 = self.conv4_2(out4 + upx2(out5))
        out3 = self.conv3_2(out3 + upx2(out4))
        out2 = self.conv2_2(out2 + upx2(out3))
        out1 = self.conv1_2(out1 + out2)
        return out1, out2, out3, out4, out5
