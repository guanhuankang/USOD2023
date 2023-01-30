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
        return x + self.conv2(ref)

class FrcPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network
    '''
    def __init__(self, dim_bin = [2048, 1024, 512, 256, 64]):
        super().__init__()
        self.frc = nn.ModuleList([ FRC(f_in, f_out) for f_in, f_out in zip(dim_bin[0:-2], dim_bin[1:-1]) ])
        self.conv1 = nn.Sequential(nn.Conv2d(dim_bin[-2], dim_bin[-1], 1), nn.BatchNorm2d(dim_bin[-1]), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(dim_bin[-1], dim_bin[-1], 1), nn.BatchNorm2d(dim_bin[-1]), nn.ReLU())

    def forward(self, features):
        ''' features: f5,f4,f3,f2,f1 '''
        n = len(features)
        out = [features[0],]
        for i in range(n-2):
            out.append(self.frc[i](out[-1], features[i+1]))
        out.append( self.conv1(out[-1])+self.conv2(features[-1]) )
        return out

class DenseFrcPN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.frc45 = FRC(2048, 1024)
        self.frc34 = FRC(1024, 512)
        self.frc35 = FRC(2048, 512)
        self.frc23 = FRC(512, 256)
        self.frc24 = FRC(1024, 256)
        self.frc25 = FRC(2048, 256)
        self.conv12 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.conv11 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU())
        self.frc13 = FRC(512, 64)
        self.frc14 = FRC(1024, 64)
        self.frc15 = FRC(2048, 64)

        weight_init(self.conv12)
        weight_init(self.conv11)
    def forward(self, features):
        f5, f4, f3, f2, f1 = features
        f4 = self.frc45(f5, f4)
        f3 = self.frc34(f4, f3) + self.frc35(f5, f3)
        f2 = self.frc23(f3, f2) + self.frc24(f4, f2) + self.frc25(f5, f2)
        f1 = self.conv12(f2) + self.conv11(f1) + self.frc13(f3, f1) + self.frc14(f4, f1) + self.frc15(f5, f1)
        return [f5, f4, f3, f2, f1]