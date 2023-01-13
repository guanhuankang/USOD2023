import torch

def pushAwayLoss(x, eps=1e-3):
    dn = torch.abs(torch.sigmoid(x) - 0.5)
    loss = -torch.log(2.0*dn+eps)
    return loss.mean()

def distributionLoss(s, t):
    ## *,n_class
    t = t.detach()
    return (-t*torch.log(s+1e-6)).sum(dim=-1).mean()

def iouLoss(pred, mask):
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3)) - inter
    iou  = 1.0-(inter+1e-6)/(union+1e-6)
    return iou.mean()

def fbetaLoss(pred, mask, beta2=0.3):
    eps = 1e-6
    tp = (pred * mask).sum(dim=(2,3))
    prc = (tp + eps) / (pred.sum(dim=(2,3)) + eps)
    rec = (tp + eps) / (mask.sum(dim=(2,3)) + eps)
    fbeta = (1.0+beta2) * (prc * rec) / ((beta2 * prc) + rec + eps)
    return 1.0 - fbeta.mean()
