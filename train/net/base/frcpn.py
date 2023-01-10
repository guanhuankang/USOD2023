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

    def forward(self, x, ref):
        ur = self.unfold(ref) ## b,c,w_s,h,w
        um = torch.mean(ur, dim=2, keepdim=True)
        pos = (ur - um).gt(0.0).float()
        ure = (torch.unsqueeze(ref, dim=2) - um).gt(0.0).float() ## stop gradient because of gt
        mask = ure * pos + (1.0 - ure) * (1.0 - pos) ## b,c,w_s,h,w
        ux = self.unfold(F.interpolate(self.conv1(x), size=ref.shape[2::], mode="nearest")) ## b,c,w_s,h,w
        x = torch.sum(ux * mask, dim=2) / (torch.sum(mask, dim=2) + 1e-6)
        return x + self.conv2(ref)

class FrcPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network
    '''
    def __init__(self, dim_bin = [2048, 1024, 512, 256, 64]):
        super().__init__()
        self.frc = nn.ModuleList([ FRC(f_in, f_out) for f_in, f_out in zip(dim_bin[0:-1], dim_bin[1::]) ])

    def forward(self, features):
        ''' features: f5,f4,f3,f2,f1 '''
        n = len(features)
        out = [features[0],]
        for i in range(n-1):
            out.append(self.frc[i](out[-1], features[i+1]))
        return out