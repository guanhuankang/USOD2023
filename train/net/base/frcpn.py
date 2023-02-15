import torch
import torch.nn as nn
import torch.nn.functional as F
from net.base.modules import UnFold, weight_init
from net.base.cbam import CBAM

class FRC(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(d_in, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(d_out, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.cbam = CBAM(d_out)
        self.unfold = UnFold(3, padding=1)
        weight_init(self)

    def calcMask(self, ref):
        ur = self.unfold(ref) ## b,c,w_s,h,w
        um = torch.mean(ur, dim=2, keepdim=True)
        pos = (ur - um).gt(0.0).float()
        ure = (torch.unsqueeze(ref, dim=2) - um).gt(0.0).float() ## stop gradient because of gt
        mask = ure * pos + (1.0 - ure) * (1.0 - pos) ## b,c,w_s,h,w
        return mask

    def forward(self, x, ref):
        mask = self.calcMask(ref)
        x = self.unfold(F.interpolate(self.conv1(x), size=ref.shape[2::], mode="nearest")) ## b,c,w_s,h,w
        x = torch.sum(x * mask, dim=2) / (torch.sum(mask, dim=2) + 1e-6)
        return self.cbam(x + self.conv2(ref))

class FrcPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network
    '''
    def __init__(self):
        super().__init__()
        self.frc4 = FRC(2048, 1024)
        self.frc3 = FRC(1024, 512)
        self.frc2 = FRC(512, 256)
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv1 = nn.Sequential(nn.Conv2d( 64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.cbam = CBAM(64)

    def forward(self, features):
        f5, f4, f3, f2, f1 = features
        f4 = self.frc4(f5, f4)
        f3 = self.frc3(f4, f3)
        f2 = self.frc2(f3, f2)
        f1 = self.cbam(self.conv2(f2) + self.conv1(f1))
        return [f5, f4, f3, f2, f1]

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv5 = nn.Sequential(nn.Conv2d(2048, 1024, 1), nn.BatchNorm2d(1024), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU())

    def forward(self, features):
        up2 = lambda x: F.interpolate(x, scale_factor=2, mode="bilinear")
        f5, f4, f3, f2, f1 = features
        f4 = up2(self.conv5(f5)) + f4
        f3 = up2(self.conv4(f4)) + f3
        f2 = up2(self.conv3(f3)) + f2
        f1 = self.conv2(f2) + f1
        return [f5, f4, f3, f2, f1]
