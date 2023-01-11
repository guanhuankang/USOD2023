#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import os
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as pth_transforms
import progressbar
import pandas as pd

class TestModel:
    def __init__(self, resultPath="output"):
        self.resultPath = resultPath
        os.makedirs(resultPath, exist_ok=True)
        self.results = []
        self.indexs = []

    def eval(self, pred, mask):
        pred = F.interpolate(pred, size=mask.shape[2::], mode="bilinear")
        assert pred.shape==mask.shape, "pred shape:{}, mask shape:{}".format(pred.shape, mask.shape)
        pos = pred.gt(0.5).float()
        tp = (pos * mask).sum()
        prc = tp / (pos.sum()+1e-6)
        rec = tp / (mask.sum()+1e-6)

        mae = torch.abs(pred-mask).mean()
        fbeta = 1.3*(prc * rec) / (0.3 * prc + rec + 1e-9)
        acc = (pos == mask).sum() / torch.numel(mask)
        iou = tp / (pos.sum() + mask.sum() - tp + 1e-6)
        gtp = mask.sum() / torch.numel(mask)
        predp = pos.sum() / torch.numel(pred)
        result = {
            "mae": mae.item(),
            "fbeta": fbeta.item(),
            "acc": acc.item(),
            "iou": iou.item(),
            "gtp": gtp.item(),
            "predp": predp.item()
        }
        return result

    def record(self, name, result):
        self.results.append(result)
        self.indexs.append(name)

    def report(self, name):
        resultPd = pd.DataFrame(self.results, index=self.indexs)
        resultPd = pd.concat([resultPd.agg("mean").to_frame(name="mean").T, resultPd], ignore_index=False)
        resultPd.to_csv(os.path.join(self.resultPath, name+".csv"))
        print(resultPd.head(1))
        return resultPd

    def test(self, tCfg, model, checkpoint=None, name="test", crf=0, save=False):
        name_now = name+"_"+str(datetime.datetime.now()).replace("-","_").replace(":","_").replace(" ","_")

        name_list = [x[0:-len(tCfg.image.suffix)] for x in os.listdir(tCfg.image.path) if x.endswith(tCfg.image.suffix)]
        transform = pth_transforms.Compose([
            pth_transforms.Resize((352,352)),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        widgets = ["[",progressbar.Timer(),"]",progressbar.Bar("*"),"(",progressbar.ETA(),")"]
        bar = progressbar.ProgressBar(maxval=len(name_list), widgets=widgets).start()
        mode = model.training
        if not isinstance(checkpoint, type(None)):
            model.loadCheckPoint(checkpoint)
        model.eval()
        model.train(False)
        with torch.no_grad():
            for i,name in enumerate(name_list[0:100]):
                img = transform(Image.open(os.path.join(tCfg.image.path, name+tCfg.image.suffix))).unsqueeze(0)
                mak = np.array(Image.open(os.path.join(tCfg.mask.path, name+tCfg.mask.suffix)).convert("L")).astype(float) / 255.0
                mak = torch.tensor(mak).unsqueeze(0).unsqueeze(0).gt(0.5).float()

                pred = model(img.cuda())["pred"].cpu()
                pred = F.interpolate(pred, size=mak.shape[2::], mode="bilinear")

                result = self.eval(pred, mak)
                self.record(name, result)
                if save:
                    out_path = os.path.join(self.resultPath, name_now)
                    os.makedirs(out_path, exist_ok=True)
                    Image.fromarray((pred[0,0].numpy()*255).astype(np.uint8)).save(os.path.join(out_path,name+".png"))
                bar.update(i)

            bar.finish()
        model.train(mode)
        return self.report(name_now)

if __name__=="__main__":
    from common import loadConfig, loadConfigByPath
    from network import Network
    cfg = loadConfig()
    tCfg = loadConfigByPath(cfg.datasetCfgPath).DUTS
    net = Network(cfg).cuda()
    testModel = TestModel()

    ckp_folder = ["A_checkpoint", "B_checkpoint"]
    ckps = [os.path.join(cf, ckp) for cf in ckp_folder for ckp in os.listdir(cf) if ckp.ednswith(".pth")]
    results = []
    for ckp in ckps:
        print(ckp, "...")
        r = testModel.test(tCfg, name=ckp, model=net, save=True, checkpoint=ckp)
        results.append( {"ckp": ckp} | r.head(1).to_dict("records")[0] )
    pd.DataFrame(results).to_csv("results.csv")
