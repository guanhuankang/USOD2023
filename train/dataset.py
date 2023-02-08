#!/usr/bin/python3
#coding=utf-8

import os, json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as pth_transforms
import albumentations as DA
from common import *

########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg, mode):
        cfg.mode = mode
        cfg.datasetCfg = loadJson(cfg.datasetCfgPath)
        datasetCfg = cfg.datasetCfg[cfg.trainSet] if mode=="train" else cfg.datasetCfg[cfg.testSet]
        datasetCfg = nestedNameSpaceFromDict(datasetCfg)
        print("datasetCfg", datasetCfg)

        self.cfg        = cfg
        self.samples = [ x[0:-4] for x in os.listdir(datasetCfg.image.path) if x.endswith(datasetCfg.image.suffix) ]
        self.datasetCfg = datasetCfg
        print("dataset length", len(self.samples), flush=True)

    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __getitem__(self, idx):
        view_transform = DA.Compose(
            [
                DA.ColorJitter(p=1.0),
                DA.RandomBrightnessContrast(p=0.5),
                DA.RGBShift()
            ]
        )

        image_transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        mask_transform = pth_transforms.Compose([
            pth_transforms.ToTensor()
        ])

        name  = self.samples[idx]
        image = Image.open(os.path.join(self.datasetCfg.image.path, name+self.datasetCfg.image.suffix)).convert("RGB")

        if self.cfg.mode=='train':
            mask = mask_transform(Image.open(os.path.join(self.datasetCfg.mask.path, name + self.datasetCfg.mask.suffix)).convert("L"))
            # sal  = mask_transform(Image.open(os.path.join(self.cfg.salPath, name + ".png")).convert("L"))

            # view_image = Image.fromarray(view_transform(image=np.array(image, dtype=np.uint8))["image"])
            # view_image_tensor = image_transform(view_image)
            image_tensor = image_transform(image)

            ## flipping
            if np.random.rand()>=0.5:
                mask = torch.flip(mask, dims=[-1])
                # sal  = torch.flip(sal , dims=[-1])
                image_tensor = torch.flip(image_tensor, dims=[-1])
                # view_image_tensor = torch.flip(view_image_tensor, dims=[-1])

            return image_tensor, image_tensor, mask
        else:
            test_transform = pth_transforms.Compose([
                pth_transforms.Resize(self.cfg.size),
                pth_transforms.ToTensor(),
                pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            t_image = test_transform(image)
            np_image = np.array(image, dtype=np.uint8)
            shape = np_image.shape[:2]
            return t_image, np_image, shape, name

    def __len__(self):
        return len(self.samples)

    def testCollate(self, batch):
        image, uint8_img, shape, name = [list(item) for item in zip(*batch)]
        return torch.stack(image,dim=0), uint8_img, shape, name

    def collate(self, batch):
        # size = [224, 256, 288, 320, 352][-1]
        size = self.cfg.size
        image0, image1, mask = [list(item) for item in zip(*batch)]

        image0 = [F.interpolate(x.unsqueeze(0), size=size, mode="bilinear") for x in image0]
        # image1 = [F.interpolate(x.unsqueeze(0), size=size, mode="bilinear") for x in image1]
        mask   = [F.interpolate(x.unsqueeze(0), size=size, mode="nearest") for x in mask]

        image = torch.cat(image0, dim=0)
        mask  = torch.cat(mask, dim=0)
        return image, mask
