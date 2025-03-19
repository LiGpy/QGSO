import torch
import math


class Diff:
    def __init__(self):
        self.parameters_names = []
        self.parameters_all = []
        self.parameters_c = []
        self.parameters_d = []
        self.parameters_n = []
        self.parameters_V = []
        self.parameters_k = []
        self.parameters_ai = []
        self.lr_basics = None
        self.optimizer_optics = None
        self.curriculum_fov = None
        self.aper_weight = None
        self.fov_weight = None
        self.curriculum_start = None
        self.curriculum_all = None
        self.weight_loss_constraint = 0
        self.weight_img = 0
        self.weight_loss_raytrace = 0
        self.weight_simulation = 0
        self.weight_rms = 0

        self.weight_color = 0
        self.weight_fov = 0
        self.weight_constraint = 0
        self.weight_price = 0

        self.weight_ray_angle = 0
        self.weight_loss_color = 0
        self.loss_optics = torch.Tensor([0])[0]
        self.loss_rms = torch.Tensor([0])[0]
        self.loss_color = torch.Tensor([0])[0]
        self.loss_raytrace = torch.Tensor([0])[0]
        self.loss_raytrace_num = torch.Tensor([0])[0]
        self.loss_constraint = torch.Tensor([0])[0]
        self.loss_fov_max = torch.Tensor([0])[0]
        self.loss_raytrace_flag = False
        self.only_fov = None
        self.only_spot = None
        self.train_rate_now = None
        self.train_rate = None
        self.datas = None
        self.grads = None
        self.surfaces = None
        self.materials = None

        self.op_n_num = None
        self.sel_1_num = None

    def set_diff_parameters(self, surfaces, materials, diff_names_c=None, diff_names_d=None, diff_names_k=None,
                            diff_names_V=None, diff_names_ai=None, diff_names_n=None):
        self.surfaces = surfaces
        self.materials = materials
        diff_names = diff_names_c + diff_names_d + diff_names_n + diff_names_V
        for name in diff_names:
            if type(name) is str:  # lens parameter name
                self.parameters_names.append(name)
                exec('self.{}.requires_grad = True'.format(name))
                exec('self.parameters_all.append(self.{})'.format(name))
        for name in diff_names_c:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_c.append(self.{})'.format(name))
        for name in diff_names_d:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_d.append(self.{})'.format(name))
        for name in diff_names_n:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_n.append(self.{})'.format(name))
        for name in diff_names_V:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_V.append(self.{})'.format(name))
        for name in diff_names_k:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_k.append(self.{})'.format(name))
        for name in diff_names_ai:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_ai.append(self.{})'.format(name))

    def set_cosine_annealing(self, iteration, lr_scale=0.1, t_max=50):
        lr = self.lr_basics
        for i in range(len(self.optimizer_optics.param_groups)):
            self.optimizer_optics.param_groups[i]['lr'] = lr[i] * lr_scale + 0.5 * (lr[i] - lr[i] * lr_scale) * (
                    1 + math.cos(iteration / t_max * math.pi))

    def init_raytrace(self, device='cuda:0'):
        self.loss_raytrace = torch.Tensor([0])[0].to(device)
        self.loss_raytrace_num = 0
        self.loss_raytrace_flag = True

    def back_raytrace(self, iteration, weight=1000):
        print('iteration:{num}'.format(num=iteration))
        print('ray_trace_not_normal')
        print('loss_raytrace: {data}'.format(data=self.loss_raytrace.item()))
        (weight * self.loss_raytrace).backward()
        self.optimizer_optics.step()
        print('replace_params: {data}'.format(data=self.parameters_all))

    def back_constraint(self, iteration):
        print('iteration:{num}'.format(num=iteration))
        print('constraint_not_normal')
        self.loss_constraint.backward()
        self.optimizer_optics.step()
        print('replace_params: {data}'.format(data=self.parameters_all))

    def calc_optics_loss(self, RMS_all, p_cen_all):
        self.loss_color = (((RMS_all[1:, 1] - RMS_all[1:, 0]) ** 2
                       + (RMS_all[1:, 1] - RMS_all[1:, 2]) ** 2
                       + (RMS_all[1:, 0] - RMS_all[1:, 2]) ** 2).mean()
                      + ((p_cen_all[1:, 1] - p_cen_all[1:, 0]).square().sum(dim=1) + (
                        p_cen_all[1:, 1] - p_cen_all[1:, 2]).square().sum(dim=1) + (
                                 p_cen_all[1:, 0] - p_cen_all[1:, 2]).square().sum(dim=1)).mean())
        self.loss_rms = RMS_all.mean()
        self.loss_optics = self.loss_rms + self.loss_color

    def step(self, PSF, d_PSF):
        (self.loss_constraint * self.weight_loss_constraint + self.loss_raytrace * self.weight_loss_raytrace
         + self.loss_rms * self.weight_loss_rms + self.loss_color * self.weight_loss_color
         + self.loss_fov_max * self.weight_max_fov_loss).backward()
        grad_img2psf = PSF.grad / self.train_rate_now * self.weight_simulation
        for i, x in enumerate(self.parameters_all):
            x.grad = x.grad + torch.sum(grad_img2psf * d_PSF[:, :, i, :, :])  # 计算梯度

        # Adam方法进行梯度更新
        self.datas = []
        self.grads = []
        self.optimizer_optics.step()
        for i, x in enumerate(self.parameters_all):
            self.datas.append(x.item())
            self.grads.append(x.grad.item())
