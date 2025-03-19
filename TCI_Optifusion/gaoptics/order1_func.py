import os
from torchvision import io


class Order1:
    def __init__(self):
        self.surface_num = None
        self.len_parameters = None
        self.phi_inf = None
        self.vari_flag = None
        self.surfaces = None
        self.materials = None
        self.material_lab = None
        self.wave_data = None
        self.z_sensor = None
        self.approx_z_sensor = None
        self.effl = None
        self.enpd = None
        self.enpp = None
        self.valid_idx = None
        self.valid_idx_now = None
        self.fov_samples = None
        self.fov_pos = None
        self.enp_xyz = None
        self.vignetting_factors = None
        self.fit_all = None
        self.final_select_flag = None
        self.acc_data = None
        self.acc_fit_all = None
        self.now_gener_num = 0
        self.now_structures = None
        self.c_list = []
        self.d_list = []
        self.n_list = []
        self.v_list = []
        self.k_list = []
        self.ai_list = []
        self.fix_data = None
        self.stage = None
        self.opti_stage = None
        self.now_gener = None
        self.lr_rate = None
        self.use_real_glass = False

        self.img_r = None
        self.zoom_rate = None
        self.wave_data = None
        self.aperture_max = None
        self.aperture_weight = None
        self.fov_max = None

        self.psf_cen_all = None
        self.PSF = None
        self.d_PSF = None
        self.psf_R_all = None
        self.raw_psf = None
        self.PSF_real_pixel_size = None
        self.loss_map = None
        self.expp = None
        self.expd = None
        self.aper_flag = None
        self.init_population = None
