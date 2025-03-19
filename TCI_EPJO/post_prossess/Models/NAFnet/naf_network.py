import math
import torch
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm
from post_prossess.Models.NAFnet.naf_unet import NAFNet
from torch import nn
from torchvision import utils


class Network(nn.Module):
    def __init__(self, naf_net):
        super().__init__()
        self.naf_fn = NAFNet(**naf_net)
        self.iter = None

    @torch.no_grad()
    def restoration(self, y_cond):
        x_pred = self.naf_fn(y_cond)
        return x_pred

    def forward(self, y_blur):
        x_init = self.naf_fn(y_blur)
        return x_init
