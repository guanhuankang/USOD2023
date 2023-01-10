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
        res = torch.mean(ref, dim=1, keepdim=True) ## b,1,h,w
        unfold_res = self.unfold(res) ## b,1,w_s,h,w
        unfold_avg = torch.mean(unfold_res, dim=2, keepdim=True) ## b,1,1,h,w
        unfold_pos = (unfold_res - unfold_avg).gt(0.0).float() ## stop gradient b,1,w_s,h,w
        unfold_ind = (torch.unsqueeze(res, dim=2) - unfold_avg).gt(0.0).float() ## b,1,1,h,w
        mask = unfold_ind * unfold_pos + (1.0 - unfold_ind) * (1.0 - unfold_pos) ## b,1,w_s,h,w
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
    def __init__(self):
        super().__init__()
        d_model = 64
        self.frc4 = FRC(2048, 1024)
        self.frc3 = FRC(1024, 512)
        self.frc2 = FRC(512, 256)
        self.conv1 = nn.Sequential(nn.Conv2d(256, d_model, 1), nn.BatchNorm2d(d_model), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d( 64, d_model, 1), nn.BatchNorm2d(d_model), nn.ReLU())

    def forward(self, features):
        f1, f2, f3, f4, f5 = features
        c4 = self.frc4(f5, f4)
        c3 = self.frc3(c4, f3)
        c2 = self.frc2(c3, f2)
        c1 = self.conv1(c2) + self.conv2(f1)
        return [c1, c2, c3, c4, f5]
