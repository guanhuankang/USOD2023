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
from progress.bar import Bar

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
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg.T_0, eta_min=cfg.eta_min)
    sw = SummaryWriter(cfg.eventPath)
    ## parameter
    global_step = 0
    clock_begin = time.time()
    tot_iter = cfg.epoch * len(loader)
    ## testmodel
    tCfg = loadConfigByPath(cfg.datasetCfgPath)
    testResults = []
    ## avg
    loss_avg = Avg()
    bce_avg = Avg()
    lwt_avg = Avg()
    cl_avg = Avg()
    attn_avg = Avg()
    pred_avg = Avg()
    mask_avg = Avg()

    for epoch in range(cfg.epoch):
        # optimizer.param_groups[0]['lr'] = (1.0 - (epoch / cfg.epoch)**0.9) * cfg.lr
        print("epoch:", epoch, " # dataset len:", len(loader), flush=True)
        net.train(True)
        bar = Bar(max=len(loader))
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

            ## avg
            loss_avg.update(loss.item())
            cl_avg.update(out["loss_dict"]["clloss"])
            bce_avg.update(out["loss_dict"]["bceloss"])
            lwt_avg.update(out["loss_dict"]["lwtloss"])
            attn_avg.update(out["attn"].mean().item())
            pred_avg.update(out["pred"].mean().item())
            mask_avg.update(mask.gt(0.5).float().mean().item())

            if step%10 == 0 or True:
                elase = time.time() - clock_begin
                remain = elase/global_step * tot_iter - elase
                s = 'epoch:{}/{} | {:1.2f}% | lr={:1.5f} | loss={:1.3f} [cl={:1.3f} bce={:1.3f} lwt={:1.3f}] | elase={:1.2f}min | remain={:1.2f}min | attn={:1.3f} | pred={:1.3f} | mask={:1.3f} progress$'.format(
                    epoch+1, cfg.epoch, global_step/tot_iter*100.0,
                    optimizer.param_groups[0]['lr'], loss_avg(), cl_avg(), bce_avg(), lwt_avg(),
                    elase / 60, remain / 60,
                    attn_avg(), pred_avg(), mask_avg()
                )
                bar.bar_prefix = s
            bar.next()

        ## epoch end/ start epoch test
        scheduler.step()

        if epoch>=0:
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