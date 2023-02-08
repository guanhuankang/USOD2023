import torch
import torch.nn as nn
import torch.nn.functional as F
from net.base.modules import UnFold, weight_init
from net.base.CBAM import CBAM

class FRC(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(d_in, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(d_out, d_out, 1), nn.BatchNorm2d(d_out), nn.ReLU())
        self.CBAM = CBAM(d_out)
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
        return self.CBAM(x + self.conv2(ref))

class FrcPN(nn.Module):
    '''
        Cross-Scale Feature Re-Coordinate Pyramid Network
    '''
    def __init__(self, **kwargs):
        super().__init__()
        self.frc5 = FRC(2048, 1024)
        self.frc4 = FRC(1024, 512)
        self.frc3 = FRC(512, 256)
        self.frc2 = FRC(256, 64)

        self.head1 = nn.Sequential(
            nn.Conv2d(64, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(256, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.head3 = nn.Sequential(
            nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )
        self.head4 = nn.Sequential(
            nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 1, 1)
        )

    def forward(self, features):
        uphw = lambda x,size: F.interpolate(x, size=size, mode="bilinear", align_corners=True)
        f1, f2, f3, f4, f5 = features

        f4 = self.frc5(f5, f4)
        f3 = self.frc4(f4, f3)
        f2 = self.frc3(f3, f2)
        f1 = self.frc2(f2, f1)

        p1 = self.head1(f1)
        p2 = self.head2(f2)
        p3 = self.head3(f3)
        p4 = self.head4(f4)

        return [p1, p2, p3, p4]
