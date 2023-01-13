import torch
import torch.nn as nn
import torch.nn.functional as F
from net.base.modules import UnFold, weight_init

class FRC(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(d_in, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(d_out, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.unfold = UnFold(3, padding=1)
        weight_init(self)

    def calcMask(self, ref):
        ur = self.unfold(ref) ## b,c=1,w_s,h,w
        um = torch.mean(ur, dim=2, keepdim=True)
        pos = (ur - um).gt(0.0).float()
        ure = (torch.unsqueeze(ref, dim=2) - um).gt(0.0).float() ## stop gradient because of gt
        mask = ure * pos + (1.0 - ure) * (1.0 - pos) ## b,c=1,w_s,h,w
        return mask

    def forward(self, x, ref):
        mask = self.calcMask(ref) ## b,c=1,w_s,h,w
        x = self.unfold(F.interpolate(self.conv1(x), size=ref.shape[2::], mode="nearest")) ## b,c,w_s,h,w
        x = torch.sum(x * mask, dim=2) / (torch.sum(mask, dim=2) + 1e-6)
        return x + self.conv2(ref)

class FrcPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network
    '''
    def __init__(self):
        super().__init__()
        self.frc_high = FRC(2048, 1024)
        self.frc_mid = FRC(1024, 512)
        self.frc_low = FRC(512, 256)
        self.conv = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.head = nn.Sequential(nn.Conv2d(64, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Conv2d(256, 1, 1))

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        o = self.frc_high(f5, f4)
        o = self.frc_mid(o, f3)
        o = self.frc_low(o, f2)
        o = self.head(self.conv(o)+f1)
        return o

