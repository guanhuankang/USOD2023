#!/usr/bin/python3
#coding=utf-8

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms as pth_transforms
from torch.utils.data import Dataset
from common import loadJson, nestedNameSpaceFromDict

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
    def __getstate__(self): return self.__dict__

    def __setstate__(self, d): self.__dict__.update(d)

    def __getitem__(self, idx):
        cfg = self.cfg

        share_transform = pth_transforms.Compose([
            pth_transforms.Resize(cfg.size),
            pth_transforms.ToTensor(),
        ])
        image_norm = pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        pixel_transform = pth_transforms.Compose(
            [
                pth_transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            ]
        )
        layout_transform = pth_transforms.Compose([
            pth_transforms.RandomResizedCrop(size=cfg.size, scale=(49.0/64.0, 1.0)),
            pth_transforms.RandomHorizontalFlip()
        ])

        name  = self.samples[idx]
        image = Image.open(os.path.join(self.datasetCfg.image.path, name+self.datasetCfg.image.suffix))
        mask =  Image.open(os.path.join(self.datasetCfg.mask.path, name+self.datasetCfg.mask.suffix)).convert("L")

        if self.cfg.mode=='train':
            i1 = share_transform(image)
            i2 = share_transform(pixel_transform(image))
            mak = share_transform(mask)
            combine = layout_transform(torch.cat([i1,i2,mak], dim=0))
            i1, i2, mak = image_norm(combine[0:3,:,:]), image_norm(combine[3:6,:,:]), combine[6:7,:,:].gt(0.5).float()
            return i1, i2, mak
        else:
            i1 = image_norm(share_transform(image))
            uimg = np.array(image).astype(np.uint8)
            shape = uimg.shape[:2]
            return i1, uimg, shape, name

    def __len__(self):
        return len(self.samples)

    def testCollate(self, batch):
        image, uint8_img, shape, name = [list(item) for item in zip(*batch)]
        return torch.stack(image,dim=0), uint8_img, shape, name

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        i1, i2, mak = [list(item) for item in zip(*batch)]
        i1 = F.interpolate(torch.stack(i1, dim=0), size=size, mode="bilinear")
        i2 = F.interpolate(torch.stack(i2, dim=0), size=size, mode="bilinear")
        mak = F.interpolate(torch.stack(mak, dim=0), size=size, mode="bilinear").gt(0.5).float()
        return torch.cat([i1, i2], dim=0), torch.cat([mak, mak], dim=0)
