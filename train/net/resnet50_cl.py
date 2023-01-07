#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

def min2D(m):
    return torch.min(torch.min(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]


def max2D(m):
    return torch.max(torch.max(m, dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0]


def upx2(m):
    b, d, h, w = m.shape
    return F.interpolate(m, size=(2 * h, 2 * w), mode="bilinear")


def minMaxNorm(m, eps=1e-12):
    return (m - min2D(m)) / (max2D(m) - min2D(m) + eps)

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
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

class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)

class ResNet(nn.Module):
    ''' ResNet50 '''
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)
        self.initialize()

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1, out2, out3, out4, out5

    def initialize(self):
        # self.load_state_dict(torch.load('../res/resnet50-19c8e357.pth'), strict=False)
        weight_init(self)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class ContrastiveLearningModule(nn.Module):
    def __init__(self, d_model, n_head,d_ff):
        super().__init__()
        self.sa_ff = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_ff)
        self.multi_head = nn.MultiheadAttention(d_model, n_head)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.initial()

    def forward(self, x):
        b,d,h,w = x.shape
        x = x.reshape(b, d, -1).permute(2, 0, 1) ## n_token, batch, d_model
        z = self.sa_ff(x)
        v = torch.mean(z, dim=0, keepdim=True)
        out, attn = self.multi_head(v, z, z) ## out: 1 * batch * d_model; attn: batch * 1 * n_token
        out = self.ff(out).squeeze(0) ## out: batch * d_model
        out = out / (torch.linalg.norm(out, dim=-1, keepdim=True) + 1e-12) ## norm: unit_ball
        # pred = (attn.reshape(b, 1, h, w) - attn.min()) / (attn.max() - attn.min() + 1e-6) ## batch * h * w
        return out, attn.reshape(b,1,h,w)

    def initial(self):
        weight_init(self)

class SumUp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, device):
        s = torch.tensor([0.0], requires_grad=True, device=device)
        for v in x:
            s = s + v
        return s

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sumUp = SumUp()

    def forward(self, x, pos_pair):
        i, j = pos_pair
        t = 0.1
        nominator = torch.exp(F.cosine_similarity(x[i].unsqueeze(0), x[j].unsqueeze(0)) / t)
        denominator = self.sumUp([torch.exp(F.cosine_similarity(x[i].unsqueeze(0), x[k].unsqueeze(0)) / t) for k in range(len(x)) if k!=i], device = x.device)
        loss = -torch.log(nominator / (denominator + 1e-6))
        return loss

class ResNet50CL(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet()
        self.conv = nn.Conv2d(2048, 512, 1)
        self.cl = ContrastiveLearningModule(512, 4, 2048)
        self.clloss = ContrastiveLoss()
        self.sumUp = SumUp()

    def initialize(self, init_weigth):
        self.load_state_dict(torch.load(init_weigth))

    def forward(self, x):
        _,_,_,_,f5 = self.backbone(x)
        ir, pred = self.cl(self.conv(f5))
        pred = F.interpolate(pred, x.shape[-2::], mode="bilinear")

        if self.training:
            n = len(x) // 2
            loss = self.sumUp([ self.clloss(ir, (i, int((i+n)%(n+n)))) for i in range(n) ], device = ir.device) / n
            return {
                "pred": minMaxNorm(pred),
                "loss": loss,
                "ir": ir
            }
        else:
            return {
                "pred": minMaxNorm(pred),
                "ir": ir
            }
