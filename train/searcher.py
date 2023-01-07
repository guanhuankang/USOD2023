import datetime
import os
import torch
from common import *
from train import train

def echoTime():
    print(datetime.datetime.now())

def go(name, cg, **kwargs):
    print("start:", datetime.datetime.now(), flush=True)
    cg.update(kwargs)
    cfg = loadConfig()

    cfg.lr = cg["lr"]
    cfg.batchSize = cg["bs"]
    cfg.epoch = cg["epoch"]
    cfg.d_model = cg["d_model"]
    cfg.d_ff = cg["d_ff"]
    cfg.name = name
    cfg.checkpointPath = "{}_checkpoint".format(name)

    torch.cuda.empty_cache()
    print(cfg)
    train(cfg)
    torch.cuda.empty_cache()

    print("done:", datetime.datetime.now(), flush=True)
    os.system("cp output.log {}_output.log".format(name))
    os.system("echo -- > output.log")

if __name__=='__main__':
    cfg = loadConfig()
    cg = {"lr": cfg.lr, "bs": cfg.batchSize, "epoch": cfg.epoch, "d_model": cfg.d_model, "d_ff": cfg.d_ff}

    go("A", cg)