import os
from torchvision import io


class Order1:
    def __init__(self):
        self.enpd = None
        self.enpp = None
        self.effl = None
        self.img_r = None
        self.zoom_rate = None
        self.wave_data = None
        self.aperture_max = None
        self.aperture_weight = None
        self.fov_max = None
        self.fov_samples = None
        self.fov_pos = None
        self.vignetting_factors = None
        self.enp_xy = None
        self.psf_cen_all = None
        self.PSF = None
        self.d_PSF = None
        self.psf_R_all = None
        self.raw_psf = None
        self.PSF_real_pixel_size = None
        self.loss_map = None
        self.expp = None
        self.expd = None
        self.exp_psf = []
