import datetime
import os

import pandas as pd
import torch
from common import *
from train import train

def echoTime():
    print(datetime.datetime.now())

def round(name, cg, **kwargs):
    print("start:", datetime.datetime.now(), flush=True)
    cg.update(kwargs)
    cfg = loadConfig()

    cfg.lr = cg["lr"]
    cfg.batchSize = cg["bs"]
    cfg.epoch = cg["epoch"]
    cfg.d_model = cg["d_model"]
    cfg.d_ff = cg["d_ff"]
    cfg.salPath = cg["salPath"]
    cfg.name = name
    cfg.checkpointPath = "{}_checkpoint".format(name)

    torch.cuda.empty_cache()
    print(cfg)
    train(cfg)
    torch.cuda.empty_cache()

    print("done:", datetime.datetime.now(), flush=True)
    os.system("cp output.log {}_output.log".format(name))
    os.system("echo -- > output.log")

def updateLabel(name, cfg, tCfg, csv=None):
    from network import Network
    from testmodel import TestModel
    net = Network(cfg).cuda()
    testModel = TestModel(resultPath="output")
    if not isinstance(csv, type(None)):
        bestep = pd.read_csv(csv).set_index("epoch")["mae"].idxmin()
    else:
        bestep = cfg.epoch
    ckp = "{name}_checkpoint/model-{ep}-{name}.pth".format(name=name, ep=bestep)
    print("Checkpoint: {ckp}".format(ckp = ckp))
    print(tCfg, flush=True)
    sname = "{name}_{ep}_DUTS_TR".format(name=name, ep=bestep)
    report = testModel.test(tCfg, name=sname, model=net, crf=1, save=True, checkpoint=ckp)
    report.to_csv("output/{sname}_results.csv".format(sname=sname))
    del net
    retPath = "output/{sname}".format(sname=sname)
    return retPath


if __name__=='__main__':
    cfg = loadConfig()
    tCfg = loadConfigByPath(cfg.datasetCfgPath).DUTS_TR
    cg = {"lr": cfg.lr, "bs": cfg.batchSize, "epoch": cfg.epoch, "d_model": cfg.d_model, "d_ff": cfg.d_ff, "salPath": cfg.salPath}

    names = ["round{}ep".format(r+1) for r in range(5)]
    for name in names:
        round(name, cg)
        # cg["salPath"] = updateLabel(name, cfg, tCfg, csv=name+"_results.csv")
        cg["salPath"] = updateLabel(name, cfg, tCfg, csv=None)


