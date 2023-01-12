#!/home/huankguan2/anaconda3/envs/cuda116/bin/python
#coding=utf-8

import cv2
from torch.utils.data import DataLoader
import tqdm, datetime
from joblib import Parallel, delayed

from common import *
from dataset import Data
from network import Network

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax

def crefRefine(img, labels, iters=5, sxy_g=3, compat_g=3, sxy_b=50, srgb_b=5, compat_b=10):
    h,w = img.shape[:2]
    d = dcrf.DenseCRF2D(w, h, 2)
    # unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)
    unary = unary_from_softmax(np.stack([labels.reshape(-1),1.0-labels.reshape(-1)], axis=0))
    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=sxy_g, compat=compat_g)
    d.addPairwiseBilateral(sxy=sxy_b, srgb=srgb_b, rgbim=img, compat=compat_b)
    infer = np.array(d.inference(iters)).astype('float32')
    res = infer[0, :].reshape(h, w)
    return res

class Test(object):
    def __init__(self, cfg):
        print(cfg)
        ## dataset
        self.cfg = cfg
        self.data   = Data(cfg, mode = "test")
        self.loader = DataLoader(self.data, batch_size=8, shuffle=False, num_workers=cfg.numWorkers, collate_fn=self.data.testCollate)
        ## network
        self.net    = Network(cfg)
        self.net.loadCheckPoint(self.cfg.snapShot)
        self.net.train(False)
        self.net.eval()
        self.net.cuda()

    def save(self):
        with torch.no_grad():
            for data in tqdm.tqdm(self.loader):
                image, uint8_img, size, name = data
                image, shape  = image.cuda().float(), size
                out = self.net(image)
                pred = [
                    torch.nn.functional.interpolate(out["pred"][i].unsqueeze(0)*255, size[i], mode="bilinear").cpu().detach().numpy().astype(np.uint8)
                        for i in range(len(image))
                ]
                uint8_img = [cv2.resize(x, tuple(reversed(size[i]))) for i,x in enumerate(uint8_img) ]

                if not os.path.exists(self.cfg.evalPath):
                    os.makedirs(self.cfg.evalPath)

                n = pred[0].shape[1]
                if cfg.crf:
                    pred = Parallel(n_jobs=8)(delayed(crefRefine)(uint8_img[bi], pred[bi][0,i].astype(float)/255.0, 1) for bi in range(len(name)) for i in range(n) )
                    pred = [(x * 255).astype(np.uint8) for x in pred]
                else:
                    pred = [ pred[bi][0,i] for bi in range(len(name)) for i in range(n)]
                names = [ name[bi] if i==0 else name[bi]+"_"+str(i) for bi in range(len(name)) for i in range(n)]
                for na, ma in zip(names, pred):
                    cv2.imwrite(os.path.join(self.cfg.evalPath, "%s.png" % na), ma)

if __name__=='__main__':
    for testSet in ["DUTS"]:
        print(datetime.datetime.now(), "running ", testSet)
        cfg = loadConfig()
        cfg.mode = "test"
        cfg.testSet = testSet
        t = Test(cfg)
        t.save()
