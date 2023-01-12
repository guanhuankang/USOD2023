import torch
from torch import nn

from copy import deepcopy
from collections import OrderedDict
from sys import stderr

from torch import Tensor


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float):
        super().__init__()
        self.decay = decay

        self.model = model
        self.shadow = deepcopy(self.model)

        for param in self.shadow.parameters():
            param.requires_grad_(False)
            param.detach_()

    def setEval(self):
        self.shadow.train(False)
        self.shadow.eval()

    @torch.no_grad()
    def update(self):
        if not self.training:
            print("EMA update should only be called during training", file=stderr, flush=True)
            return
        model_params = OrderedDict(self.model.named_parameters())
        shadow_params = OrderedDict(self.shadow.named_parameters())
        assert model_params.keys() == shadow_params.keys()
        for name, param in model_params.items():
            shadow_params[name].sub_((1. - self.decay) * (shadow_params[name] - param))

        model_buffers = OrderedDict(self.model.named_buffers())
        shadow_buffers = OrderedDict(self.shadow.named_buffers())
        assert model_buffers.keys() == shadow_buffers.keys()
        for name, buffer in model_buffers.items():
            shadow_buffers[name].copy_(buffer)

    def forward(self, x, mva=False, **kwargs) -> Tensor:
        if not mva:
            return self.model(x, **kwargs)
        else:
            return self.shadow(x, infer=True, **kwargs)
