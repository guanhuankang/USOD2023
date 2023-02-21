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
        image_transform = pth_transforms.Compose([
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        mask_transform = pth_transforms.Compose([
            pth_transforms.ToTensor()
        ])

        name  = self.samples[idx]
        image = Image.open(os.path.join(self.datasetCfg.image.path, name+self.datasetCfg.image.suffix)).convert("RGB")
        mask = Image.open(os.path.join(self.datasetCfg.mask.path, name + self.datasetCfg.mask.suffix)).convert("L")

        if self.cfg.mode=='train':
            w, h = mask.size
            getRandSize = lambda a, r: int(a * (np.random.rand()*(1-r)+r))

            aug_transform = DA.Compose(
                [
                    DA.HorizontalFlip(p=0.5),
                    DA.RandomCrop(getRandSize(h, 0.8), getRandSize(w, 0.8))
                ]
            )

            aug = aug_transform(image=np.array(image, dtype=np.uint8), mask=np.array(mask, dtype=np.uint8))
            img0, mak0 = image_transform(Image.fromarray(aug["image"])), mask_transform(Image.fromarray(aug["mask"]))

            new_aug_transform = DA.Compose(
                [
                    DA.PixelDropout(dropout_prob=0.15, drop_value=127, p=0.5),
                    DA.ToGray(p=0.5)
                ]
            )
            new_aug = new_aug_transform(image=np.array(image, dtype=np.uint8), mask=np.array(mask, dtype=np.uint8))
            img1, mak1 = image_transform(Image.fromarray(new_aug["image"])), mask_transform(Image.fromarray(new_aug["mask"]))

            return img0, img1, mak0, mak1
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
        size = self.cfg.size
        image0, image1, mask0, mask1 = [list(item) for item in zip(*batch)]

        image0 = [F.interpolate(x.unsqueeze(0), size=size, mode="bilinear") for x in image0]
        image1 = [F.interpolate(x.unsqueeze(0), size=size, mode="bilinear") for x in image1]
        mask0   = [F.interpolate(x.unsqueeze(0), size=size, mode="nearest") for x in mask0]
        mask1   = [F.interpolate(x.unsqueeze(0), size=size, mode="nearest") for x in mask1]

        image = torch.cat(image0+image1, dim=0)
        mask  = torch.cat(mask0+mask1, dim=0)
        return image, mask
