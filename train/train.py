#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import datetime
import time, os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.enabled = False

from dataset import Data
from common import *
from network import Network
from loader import Loader

def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def train(cfg):
    cfg.mode = "train"
    print(cfg)
    ## dataset
    data   = Data(cfg, mode="train")
    print("preparing loader...", flush=True)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batchSize, shuffle=True, pin_memory=True, num_workers=cfg.numWorkers)
    # loader = Loader(data, collate_fn=data.collate, batch_size=cfg.batchSize, shuffle=True, pin_memory=True, num_workers=cfg.numWorkers)
    ## network
    net    = Network(cfg)
    net.train(True)
    net.cuda()
    ## optimizer & logger
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weightDecay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epoch_lr_delay)
    sw = SummaryWriter(cfg.eventPath)
    ## parameter
    global_step = 0
    clock_begin = time.time()
    tot_iter = cfg.epoch * len(loader)

    for epoch in range(cfg.epoch):
        # optimizer.param_groups[0]['lr'] = (1.0 - (epoch / cfg.epoch)**0.9) * cfg.lr
        print("epoch:", epoch, " # dataset len:", len(loader), flush=True)
        for step, (image, mask) in enumerate(loader):
            optimizer.zero_grad()
            image, mask = image.cuda().float(), mask.cuda().float()
            out = net(image, global_step=global_step/tot_iter, sw=sw)
            loss = out["loss"]

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            ## log
            global_step += 1
            sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step/tot_iter)
            sw.add_scalars('loss', {"visible_loss":loss.item()}, global_step=global_step/tot_iter)
            if step%10 == 0:
                elase = time.time() - clock_begin
                remain = elase/global_step * tot_iter - elase
                print('%s | %.2f%% | step:%d/%d/%d | lr=%.6f | loss=%.6f | elase=%.1fmin | remain=%.1fmin'
                    %(datetime.datetime.now(), global_step/tot_iter*100.0, global_step, epoch+1, cfg.epoch, optimizer.param_groups[0]['lr'], loss.item(),
                      elase / 60, remain / 60), flush=True
                )
        ## epoch end
        scheduler.step()
        if epoch > cfg.epoch * 0.80 or epoch == int(cfg.epoch//2) or True:
            if not os.path.exists(cfg.checkpointPath): os.makedirs(cfg.checkpointPath)
            torch.save(net.state_dict(), os.path.join(cfg.checkpointPath, "model-{}-{}.pth".format(epoch+1, cfg.name)))

if __name__=='__main__':
    print(datetime.datetime.now(), "train starts training")
    cfg = loadConfig()
    train(cfg)