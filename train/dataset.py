#!/usr/bin/python3
#coding=utf-8

import os, json
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as DA
from common import *

########################### Data Augmentation ###########################
class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean 
        self.std  = std
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __call__(self, images, maps=[]):
        images = [(img - self.mean)/self.std for img in images]
        if len(maps) <= 0:
            return images
        maps = [m/255 for m in maps]
        return tuple(images), tuple(maps)

class RandomCrop(object):
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __call__(self, images, maps=[]):
        image = images[0]
        H,W,_   = image.shape
        randw   = np.random.randint(W/8)
        randh   = np.random.randint(H/8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H+offseth-randh, offsetw, W+offsetw-randw
        images = [image[p0:p1,p2:p3, :] for image in images]
        if len(maps) <= 0:
            return images
        maps = [m[p0:p1,p2:p3] for m in maps]
        return tuple(images), tuple(maps)

class RandomFlip(object):
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __call__(self, images, maps=[]):
        if np.random.randint(2)==0:
            images = [image[:,::-1,:].copy() for image in images]
            if len(maps) <= 0:
                return images
            maps = [m[:, ::-1].copy() for m in maps]
            return tuple(images), tuple(maps)
        else:
            if len(maps) <= 0:
                return images
            return tuple(images), tuple(maps)

class Resize(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __call__(self, images, maps=[]):
        images = [ cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR) for image in images]
        if len(maps) <= 0:
            return images
        maps = [cv2.resize( m, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR) for m in maps]
        return tuple(images), tuple(maps)

class ToTensor(object):
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __call__(self, image, *maps):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        if len(maps) <= 0:
            return image
        maps = [image] + [ torch.from_numpy(m) for m in maps]
        return tuple(maps)

########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg, mode):
        cfg.mode = mode
        cfg.datasetCfg = loadJson(cfg.datasetCfgPath)
        datasetCfg = cfg.datasetCfg[cfg.trainSet] if mode=="train" else cfg.datasetCfg[cfg.testSet]
        datasetCfg = nestedNameSpaceFromDict(datasetCfg)
        print("datasetCfg", datasetCfg)

        self.cfg        = cfg
        self.normalize  = Normalize(mean=cfg.mean, std=cfg.std)
        self.randomcrop = RandomCrop()
        self.randomflip = RandomFlip()
        self.resize     = Resize(*cfg.size)
        self.totensor   = ToTensor()
        self.samples = [ x[0:-4] for x in os.listdir(datasetCfg.image.path) if x.endswith(datasetCfg.image.suffix) ]
        self.datasetCfg = datasetCfg
    def __getstate__(self): return self.__dict__
    def __setstate__(self, d): self.__dict__.update(d)
    def __getitem__(self, idx):
        transform = DA.Compose(
            [
                DA.ColorJitter(p=1.0),
                DA.RandomBrightnessContrast(p=0.5),
                DA.RGBShift()
            ]
        )

        name  = self.samples[idx]
        image = cv2.imread(os.path.join(self.datasetCfg.image.path, name+self.datasetCfg.image.suffix))[:,:,::-1].astype(np.uint8) ## RGB

        if self.cfg.mode=='train':
            # mask = cv2.imread(os.path.join(self.datasetCfg.mask.path, name + self.datasetCfg.mask.suffix), 0).astype(
            #     np.float32)
            # sal  = cv2.imread(os.path.join(salCfg.path, name+salCfg.suffix), 0).astype(np.float32)
            mask = np.zeros_like(image)[:,:,0] ## fake mask

            images, masks = self.normalize([image.astype(np.float32), transform(image=image)["image"].astype(np.float32)], [mask])
            images, masks = self.randomcrop(images, masks)
            images, masks = self.randomflip(images, masks)
            return images[0], images[1], masks[0]
        else:
            shape = image.shape[:2]
            uint8_img = image.copy()

            images = self.normalize([image.astype(np.float32)])
            image = self.resize(images)[0]
            image = self.totensor(image)
            return image, uint8_img, shape, name

    def __len__(self):
        return len(self.samples)

    def testCollate(self, batch):
        image, uint8_img, shape, name = [list(item) for item in zip(*batch)]
        return torch.stack(image,dim=0), uint8_img, shape, name

    def collate(self, batch):
        size = [224, 256, 288, 320, 352][np.random.randint(0, 5)]
        image0, image1, mask = [list(item) for item in zip(*batch)]
        for i in range(len(batch)):
            image0[i] = cv2.resize(image0[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            image1[i] = cv2.resize(image1[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i]  = cv2.resize(mask[i],  dsize=(size, size), interpolation=cv2.INTER_LINEAR)
        image  = torch.from_numpy(np.stack(image0 + image1, axis=0)).permute(0,3,1,2)
        mask   = torch.from_numpy(np.stack(mask + mask, axis=0)).unsqueeze(1)
        return image, mask
