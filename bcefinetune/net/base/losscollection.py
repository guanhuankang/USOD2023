import torch

def pushAwayLoss(x, eps=1e-3):
    dn = torch.abs(torch.sigmoid(x) - 0.5)
    loss = -torch.log(2.0*dn+eps)
    return loss.mean()

def distributionLoss(s, t):
    ## *,n_class
    t = t.detach()
    return (-t*torch.log(s+1e-6)).sum(dim=-1).mean()