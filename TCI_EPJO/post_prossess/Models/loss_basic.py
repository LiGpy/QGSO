# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
import lpips
from torch import nn as nn
from torch.nn import functional as F
import numpy as np


def l1_loss(pred, target):
    return F.l1_loss(pred, target)


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def psnr_loss(pred, target, weight=1):
    return weight * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()


class LpipsLoss:
    def __init__(self, version=0.1, device='cuda:0'):
        self.loss_fn = lpips.LPIPS(net='alex', version=version).to(device)

    def calc(self, pred, target, weight=1):
        current_lpips_distance = weight * self.loss_fn.forward(pred, target)
        return current_lpips_distance


class WeightLoss:
    def __init__(self, version=0.1, device='cuda:0', lpips_weight=0.5):
        self.loss_lpips = lpips.LPIPS(net='alex', version=version).to(device)
        self.lpips_weight = lpips_weight

    def calc(self, pred, target, loss_weight=1):
        current_lpips_distance = self.loss_lpips.forward(pred, target)
        return loss_weight * (self.lpips_weight * current_lpips_distance + (1 - self.lpips_weight) * mse_loss(pred, target))
