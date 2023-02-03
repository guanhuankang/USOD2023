#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
# coding=utf-8
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

from structureawaretool import SaliencyStructureConsistency
# from net.r50frcpn import R50FrcPN
from net.ft import FT

class Network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = FT(cfg)

    def loadCheckPoint(self, init_weight):
        self.load_state_dict(torch.load(init_weight))

    def forward(self, x, **kwargs):
        out1 = self.model(x, **kwargs)

        x_scale = F.interpolate(x, scale_factor=0.3, mode="bilinear", align_corners=True)
        out2 = self.model(x_scale, **kwargs)
        y_scale = out2["pred"]

        ref_scale = F.interpolate(out1["pred"], scale_factor=0.3, mode="bilinear", align_corners=True)
        loss_ssc = SaliencyStructureConsistency(y_scale, ref_scale, 0.85)

        loss = out1["loss"] + out2["loss"] + loss_ssc

        if "sw" in kwargs:
            kwargs["sw"].add_scalars("loss", {
                "ms": loss_ssc.item(),
                "bce1": out1["loss"].item(),
                "bce2": out2["loss"].item()
            }, global_step=kwargs["global_step"])
        return {
            "loss": loss,
            "pred": out1["pred"],
            "pred_scale": y_scale
        }


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms as pth_transforms
    from common import loadConfig, dumpPickle

    transform = pth_transforms.Compose([
        pth_transforms.Resize((352, 352)),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    names = ["ILSVRC2012_test_00000003.jpg", "ILSVRC2012_test_00000025.jpg", "ILSVRC2012_test_00000023.jpg",
             "ILSVRC2012_test_00000086.jpg"]
    imgs = [Image.open(r"D:\2023ICCVUSOD\dataset\DUTS\DUTS-TE\DUTS-TE-Image\{}".format(x)) for x in names]
    x = torch.stack([transform(img) for img in imgs], dim=0)

    cfg = loadConfig()
    net = Network(cfg)
    # net.loadCheckPoint(cfg.snapShot)
    net.train()
    # net.eval()
    out = net(x, global_step = 0.101)
    print("loss",out["loss"])
    print("pred", out["pred"].shape)