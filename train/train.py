#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import datetime
import time, os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
torch.backends.cudnn.enabled = False

from dataset import Data
from common import *
from network import Network
from loader import Loader
from testmodel import TestModel

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
    base, building = [], []
    for name, param in net.named_parameters():
        if "backbone.conv1" in name or "backbone.bn1" in name:
            continue
        elif "backbone" in name:
            base.append(param)
        else:
            building.append(param)
    optimizer = torch.optim.SGD([{"params": base}, {"params": building}], lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weightDecay) ## net.parameters()

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.epoch_lr_delay)
    sw = SummaryWriter(cfg.eventPath)
    ## parameter
    global_step = 0
    clock_begin = time.time()
    tot_iter = cfg.epoch * len(loader)
    ## testmodel
    tCfg = loadConfigByPath(cfg.datasetCfgPath)
    testResults = []

    for epoch in range(cfg.epoch):
        eppercent = min(1.0,max(0.0, epoch - cfg.epoch_lr_delay)) / min(1.0, max(1e-6, cfg.epoch - cfg.epoch_lr_delay))
        optimizer.param_groups[0]['lr'] = (1.0 - eppercent**0.9) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1.0 - eppercent**0.9) * cfg.lr

        print("epoch:", epoch, " # dataset len:", len(loader), flush=True)
        net.train(True)
        for step, (image, mask) in enumerate(loader):
            optimizer.zero_grad()
            image, mask = image.cuda().float(), mask.cuda().float()
            out = net(image, global_step=global_step/tot_iter, sw=sw, epoches=cfg.epoch)
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
                    %(datetime.datetime.now(), global_step/tot_iter*100.0, global_step, epoch+1, cfg.epoch, optimizer.param_groups[1]['lr'], loss.item(),
                      elase / 60, remain / 60), flush=True
                )
        ## epoch end/ start epoch test
        # scheduler.step()
        if epoch > cfg.epoch * 0.80 or epoch == int(cfg.epoch//2) or epoch>6:
            if not os.path.exists(cfg.checkpointPath): os.makedirs(cfg.checkpointPath)
            torch.save(net.state_dict(), os.path.join(cfg.checkpointPath, "model-{}-{}.pth".format(epoch+1, cfg.name)))
            with torch.no_grad():
                r = TestModel().test(tCfg=tCfg.DUTS, model=net, name=cfg.name+str(epoch+1), checkpoint=None, crf=0, save=False)
                sw.add_scalars("val", r.head(1).to_dict("records")[0], global_step=epoch)
                testResults.append({"epoch": epoch+1, "name": cfg.name} | r.head(1).to_dict("records")[0])
                print(pd.DataFrame(testResults).set_index("epoch").sort_index(), flush=True)
        ## end epoch test
    testResults = pd.DataFrame(testResults).set_index("epoch").sort_index()
    print(testResults, flush=True)
    testResults.to_csv(cfg.name+"_results.csv")

if __name__=='__main__':
    print(datetime.datetime.now(), "train starts training")
    cfg = loadConfig()
    train(cfg)