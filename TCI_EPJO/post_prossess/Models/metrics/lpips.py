import torch
from torch.nn import functional as F
import lpips


class calculate_lpips:
    def __init__(self,  version=0.1, device='cuda:0'):
        self.loss_fn = lpips.LPIPS(net='alex', version=version).to(device)

    def calc(self, pred, target, weight=1):
        current_lpips_distance = weight * self.loss_fn.forward(pred / 0.5 - 1, target / 0.5 - 1)
        return current_lpips_distance

