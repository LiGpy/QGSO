import os
import time

import torch
import math
from .basics import *
import numpy as np
from torchvision import io
from torchvision import utils
from post_prossess.Models.metrics import calculate_psnr, calculate_ssim, calculate_lpips
from post_prossess.Models.loss_basic import mse_loss, l1_loss, LpipsLoss, WeightLoss
from torch.nn import functional as F


class Diff:
    def __init__(self):
        self.final_loss = None
        self.net_name = None
        self.period_thi = None
        self.stop_flag = False
        self.period_meter = None
        self.period_test_plate = None
        self.last_epochs = None
        self.device = None
        self.length = None
        self.param_groups = None
        self.z_sensor = None
        self.lr_all = None
        self.calc_lpips = None
        self.model_root = None
        self.val_per_epochs = None
        self.test_iter = None
        self.post_iter = None
        self.parameters_names = []
        self.parameters_all = []
        self.parameters_c = []
        self.parameters_d = []
        self.parameters_n = []
        self.parameters_V = []
        self.parameters_k = []
        self.parameters_ai = []
        self.parameters_sensor = []
        self.lr_basics = None
        self.optimizer_optics = None
        self.curriculum_fov = None
        self.aper_weight = None
        self.fov_weight = None
        self.curriculum_start = None
        self.curriculum_all = None
        self.weight_loss_constraint = 0
        self.weight_loss_network = None
        self.weight_img = 0
        self.weight_rms = 0
        self.weight_ray_angle = 0
        self.weight_psf = 0
        self.weight_loss_rms = 0
        self.weight_loss_color = 0
        self.loss_optics = torch.Tensor([0])[0]
        self.loss_rms = torch.Tensor([0])[0]
        self.loss_color = torch.Tensor([0])[0]
        self.loss_raytrace = torch.Tensor([0])[0]
        self.loss_raytrace_num = torch.Tensor([0])[0]
        self.loss_constraint = torch.Tensor([0])[0]
        self.loss_fov_max = torch.Tensor([0])[0]
        self.loss_raytrace_flag = None
        self.only_fov = None
        self.only_spot = None
        self.train_rate = None
        self.datas = None
        self.data_all = []
        self.loss_all = []
        self.grads = None
        self.surfaces = None
        self.materials = None
        self.epochs = None
        self.pre_epochs = None
        self.now_epochs = 0
        self.look_per_epochs = None
        self.post_b_size = None
        self.post_img_size = None
        self.model = None
        self.writer = None
        self.flag_use_post = None
        self.constraint_all = None
        self.loss_ray_edge = None
        self.m = None
        self.v = None
        self.t = None
        self.frozen_flag = None
        self.period_all = None
        self.ao_test_plate = None
        self.tu_test_plate = None
        self.ao_path = None
        self.tu_path = None
        self.mater_name = []
        self.test_plate_order = []
        self.require_grad = None
        self.simu_rate = None


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

    def set_diff_parameters_all(self, lens):
        # 优化变量和优化器定义，默认全变量
        surfaces = lens.basics.surfaces
        materials = lens.basics.materials
        self.surfaces = surfaces
        self.materials = materials
        self.z_sensor = lens.basics.z_sensor
        diff_names_c = []
        diff_names_d = []
        diff_names_n = []
        diff_names_V = []
        diff_names_k = []
        diff_names_ai = []
        diff_name_sensor = []
        aspheric_list = []
        for ii in range(len(lens.basics.surfaces)):
            if ii != lens.basics.aper_index:
                diff_names_c += ['surfaces[{}].c'.format(ii)]
            if ii != 0:
                diff_names_d += ['surfaces[{}].d'.format(ii)]
            if lens.basics.materials[ii + 1].n > 1.0003:
                diff_names_n += ['materials[{}].n'.format(ii + 1)]
                diff_names_V += ['materials[{}].V'.format(ii + 1)]
            if ii in aspheric_list:
                diff_names_k += ['surfaces[{}].k'.format(ii)]
                for iii in range(3):
                    diff_names_ai += ['surfaces[{ii}].ai[{iii}]'.format(ii=ii, iii=iii)]
        diff_name_sensor += ['z_sensor']

        diff_names = diff_names_c + diff_names_d + diff_names_n + diff_names_V + diff_name_sensor
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
        for name in diff_name_sensor:
            if type(name) is str:  # lens parameter name
                exec('self.parameters_sensor.append(self.{})'.format(name))

    def set_optimizer_optics(self, name, beta=None, lr=None, lens=None):
        self.lr_basics = lr
        self.lr_all = []
        self.constraint_all = []
        self.param_groups = []
        constraint = lens.constraint
        materials = lens.basics.materials
        if name == 'Adam':
            for i in range(len(self.parameters_c)):
                self.param_groups.append({'params': self.parameters_c[i], 'lr': lr[0], 'betas': beta})
                self.constraint_all.append(constraint.c)
                self.lr_all.append(lr[0])
            for i in range(len(self.parameters_d)):
                self.param_groups.append({'params': self.parameters_d[i], 'lr': lr[1], 'betas': beta})
                if (materials[i].n < 1.0003 < materials[i + 1].n) or (
                        materials[i].n > 1.0003 and materials[i + 1].n > 1.0003):
                    self.constraint_all.append(constraint.thi_glass_cen)
                elif 0 <= lens.basics.aper_index - i <= 1:
                    self.constraint_all.append(constraint.aper_d)
                else:
                    self.constraint_all.append(constraint.thi_air_cen)
                self.lr_all.append(lr[1])
            for i in range(len(self.parameters_n)):
                self.param_groups.append({'params': self.parameters_n[i], 'lr': lr[2], 'betas': beta})
                self.constraint_all.append(constraint.material_n)
                self.lr_all.append(lr[2])
            for i in range(len(self.parameters_V)):
                self.param_groups.append({'params': self.parameters_V[i], 'lr': lr[3], 'betas': beta})
                self.constraint_all.append(constraint.material_V)
                self.lr_all.append(lr[3])
            for i in range(len(self.parameters_sensor)):
                self.param_groups.append({'params': self.parameters_sensor[i], 'lr': lr[1] * 10, 'betas': beta})
                self.constraint_all.append(constraint.z_sensor)
                self.lr_all.append(lr[1])
            self.adam_init()
        if name == 'SGD':
            for i in range(len(self.parameters_c)):
                self.param_groups.append({'params': self.parameters_c[i], 'lr': lr[0]})
                self.constraint_all.append(constraint.c)
                self.lr_all.append(lr[0])
            for i in range(len(self.parameters_d)):
                self.param_groups.append({'params': self.parameters_d[i], 'lr': lr[1]})
                self.constraint_all.append((0, 999))
                self.lr_all.append(lr[1])
            for i in range(len(self.parameters_n)):
                self.param_groups.append({'params': self.parameters_n[i], 'lr': lr[2]})
                self.constraint_all.append(constraint.material_n)
                self.lr_all.append(lr[2])
            for i in range(len(self.parameters_V)):
                self.param_groups.append({'params': self.parameters_V[i], 'lr': lr[3]})
                self.constraint_all.append(constraint.material_V)
                self.lr_all.append(lr[3])
            for i in range(len(self.parameters_sensor)):
                self.param_groups.append({'params': self.parameters_sensor[i], 'lr': lr[1] * 10})
                self.constraint_all.append(constraint.z_sensor)
                self.lr_all.append(lr[1] * 10)
            self.optimizer_optics = torch.optim.SGD(self.param_groups)

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

    def set_period(self, spot_convergence=-0.0002, materials=None, surfaces=None, aper_idx=None, round_num=1):
        no_aper_surfaces = surfaces.copy()
        no_aper_surfaces.pop(aper_idx)

        if self.now_epochs == 0:
            self.period_all = 'start'
            self.last_epochs = 0
            return

        if self.now_epochs > 0 and self.now_epochs % self.look_per_epochs == 0:
            if self.period_all == 'start' and (self.now_epochs - self.last_epochs) / self.look_per_epochs >= 2:
                loss_all = torch.Tensor(self.loss_all).to(self.device)[self.last_epochs:]
                if loss_all[-self.look_per_epochs:].min() - loss_all[-2 * self.look_per_epochs:-1 * self.look_per_epochs].min() > spot_convergence:
                    self.period_all = 'material'
                    self.period_meter = 0
                    self.look_per_epochs -= 1
                    print('period:' + self.period_all + '_' + str(self.period_meter))
                    select_idx = loss_all.argmin()
                    select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
                    for i in range(len(self.parameters_all)):
                        self.parameters_all[i].data = select_data[i].data
                        # self.m[i] = 0
                        # self.v[i] = 0
                        # self.t[i] = 0
                    n_all = materials[0].n_all
                    V_all = materials[0].V_all
                    meter_start_idx = len(self.parameters_c) + len(self.parameters_d)
                    now_meter = self.parameters_all[meter_start_idx:-1]
                    now_n = now_meter[self.period_meter]
                    now_V = now_meter[len(now_meter) // 2 + self.period_meter]
                    idx = (((now_n - n_all) / 0.1) ** 2 + ((now_V - V_all) / 50) ** 2).argmin()
                    now_n.data = n_all[idx].data
                    now_V.data = V_all[idx].data
                    self.mater_name.append(materials[0].name_all[idx])
                    self.frozen_flag[meter_start_idx + self.period_meter] = True
                    self.frozen_flag[meter_start_idx + self.period_meter + len(now_meter) // 2] = True
                    # 更新材料
                    self.last_epochs = self.now_epochs
            elif self.period_all == 'material' and (self.now_epochs - self.last_epochs) / self.look_per_epochs >= 2:
                loss_all = torch.Tensor(self.loss_all).to(self.device)[self.last_epochs:]
                if loss_all[-self.look_per_epochs:].min() - loss_all[-2 * self.look_per_epochs:-1 * self.look_per_epochs].min() > spot_convergence:
                    self.period_meter += 1
                    if self.period_meter == len(self.parameters_all[len(self.parameters_c) + len(self.parameters_d):-1]) // 2:
                        self.look_per_epochs -= 1
                        select_idx = loss_all.argmin()
                        select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
                        for i in range(len(self.parameters_all)):
                            self.parameters_all[i].data = select_data[i].data
                            self.parameters_all[i].requires_grad = False
                        self.flag_use_post = True
                        self.period_all = 'post'
                        self.last_epochs = self.now_epochs
                        return
                # if loss_all[-self.look_per_epochs:].min() - loss_all[-2 * self.look_per_epochs:-1 * self.look_per_epochs].min() > spot_convergence:
                #     self.period_meter += 1
                #     if self.period_meter == len(self.parameters_all[len(self.parameters_c) + len(self.parameters_d):-1]) // 2:
                #         self.period_all = 'test_plate'
                #         self.period_test_plate = 0
                #         self.look_per_epochs -= 1
                #         print('period:' + self.period_all + '_' + str(self.period_test_plate))
                #         select_idx = loss_all.argmin()
                #         select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
                #         for i in range(len(self.parameters_all)):
                #             self.parameters_all[i].data = select_data[i].data
                #
                #         self.tu_test_plate = torch.Tensor(scio.loadmat(self.tu_path)['tu_data_selc']).to(
                #             self.device).abs()
                #         self.ao_test_plate = torch.Tensor(scio.loadmat(self.ao_path)['ao_data_selc']).to(
                #             self.device).abs()
                #         c_now = self.parameters_c[self.period_test_plate]
                #         r_now = no_aper_surfaces[self.period_test_plate].r
                #         if c_now > 0:
                #             idx = (1 / c_now - self.tu_test_plate[:, 1]).abs().sort()[1]
                #             # sort_data = (1 / c_now - self.tu_test_plate[:, 1]).abs().sort()[0]
                #             sort_idx = self.tu_test_plate[:, 0][idx]
                #             sort_r = self.tu_test_plate[:, 2][idx]
                #             for i_plate in range(len(sort_idx)):
                #                 if 2 * r_now <= sort_r[i_plate] - 4:
                #                     c_now.data = (1 / self.tu_test_plate[:, 1][idx][i_plate]).data
                #                     self.test_plate_order.append('tu' + str(int(sort_idx[0].item())))
                #                     break
                #         else:
                #             idx = (1 / c_now.abs() - self.ao_test_plate[:, 1]).abs().sort()[1]
                #             # sort_data = (1 / c_now - self.tu_test_plate[:, 1]).abs().sort()[0]
                #             sort_idx = self.ao_test_plate[:, 0][idx]
                #             sort_r = self.ao_test_plate[:, 2][idx]
                #             for i_plate in range(len(sort_idx)):
                #                 if 2 * r_now <= sort_r[i_plate] - 4:
                #                     c_now.data = - (1 / self.ao_test_plate[:, 1][idx][i_plate]).data
                #                     self.test_plate_order.append('ao' + str(int(sort_idx[0].item())))
                #                     break
                #         self.frozen_flag[self.period_test_plate] = True
                #         self.last_epochs = self.now_epochs
                #         # self.stop_flag = True
                #         return
                    print('period:' + self.period_all + '_' + str(self.period_meter))
                    select_idx = loss_all.argmin()
                    select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
                    for i in range(len(self.parameters_all)):
                        self.parameters_all[i].data = select_data[i].data
                        # self.m[i] = 0
                        # self.v[i] = 0
                        # self.t[i] = 0
                    n_all = materials[0].n_all
                    V_all = materials[0].V_all
                    meter_start_idx = len(self.parameters_c) + len(self.parameters_d)
                    now_meter = self.parameters_all[meter_start_idx:-1]
                    now_n = now_meter[self.period_meter]
                    now_V = now_meter[len(now_meter) // 2 + self.period_meter]
                    idx = (((now_n - n_all) / 0.1) ** 2 + ((now_V - V_all) / 50) ** 2).argmin()
                    now_n.data = n_all[idx].data
                    now_V.data = V_all[idx].data
                    self.mater_name.append(materials[0].name_all[idx])
                    self.frozen_flag[meter_start_idx + self.period_meter] = True
                    self.frozen_flag[meter_start_idx + self.period_meter + len(now_meter) // 2] = True
                    self.last_epochs = self.now_epochs
            # elif self.period_all == 'test_plate' and (self.now_epochs - self.last_epochs) / self.look_per_epochs >= 2:
            #     loss_all = torch.Tensor(self.loss_all).to(self.device)[self.last_epochs:]
            #     if loss_all[-self.look_per_epochs:].min() - loss_all[-2 * self.look_per_epochs:-1 * self.look_per_epochs].min() > spot_convergence:
            #         self.period_test_plate += 1
            #         if self.period_test_plate == len(self.parameters_c):
            #             self.period_all = 'thi'
            #             self.period_thi = 0
            #             self.look_per_epochs -= 1
            #             print('period:' + self.period_all + '_' + str(self.period_thi))
            #             select_idx = loss_all.argmin()
            #             select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
            #             for i in range(len(self.parameters_all)):
            #                 self.parameters_all[i].data = select_data[i].data
            #             d_now = self.parameters_d[self.period_thi]
            #             d_now.data = torch.Tensor([round(d_now.data.item(), round_num)]).to(self.device)[0]
            #
            #             self.frozen_flag[self.period_thi + len(self.parameters_c)] = True
            #             self.last_epochs = self.now_epochs
            #             # self.stop_flag = True
            #             return
            #         print('period:' + self.period_all + '_' + str(self.period_test_plate))
            #         select_idx = loss_all.argmin()
            #         select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
            #         for i in range(len(self.parameters_all)):
            #             self.parameters_all[i].data = select_data[i].data
            #         c_now = self.parameters_c[self.period_test_plate]
            #         r_now = no_aper_surfaces[self.period_test_plate].r
            #         if c_now > 0:
            #             idx = (1 / c_now - self.tu_test_plate[:, 1]).abs().sort()[1]
            #             # sort_data = (1 / c_now - self.tu_test_plate[:, 1]).abs().sort()[0]
            #             sort_idx = self.tu_test_plate[:, 0][idx]
            #             sort_r = self.tu_test_plate[:, 2][idx]
            #             for i_plate in range(len(sort_idx)):
            #                 if 2 * r_now <= sort_r[i_plate] - 4:
            #                     c_now.data = (1 / self.tu_test_plate[:, 1][idx][i_plate]).data
            #                     self.test_plate_order.append('tu' + str(int(sort_idx[0].item())))
            #                     break
            #         else:
            #             idx = (1 / c_now.abs() - self.ao_test_plate[:, 1]).abs().sort()[1]
            #             # sort_data = (1 / c_now - self.tu_test_plate[:, 1]).abs().sort()[0]
            #             sort_idx = self.ao_test_plate[:, 0][idx]
            #             sort_r = self.ao_test_plate[:, 2][idx]
            #             for i_plate in range(len(sort_idx)):
            #                 if 2 * r_now <= sort_r[i_plate] - 4:
            #                     c_now.data = - (1 / self.ao_test_plate[:, 1][idx][i_plate]).data
            #                     self.test_plate_order.append('ao' + str(int(sort_idx[0].item())))
            #                     break
            #         self.frozen_flag[self.period_test_plate] = True
            #         self.last_epochs = self.now_epochs
            # elif self.period_all == 'thi' and (self.now_epochs - self.last_epochs) / self.look_per_epochs >= 2:
            #     loss_all = torch.Tensor(self.loss_all).to(self.device)[self.last_epochs:]
            #     if loss_all[-self.look_per_epochs:].min() - loss_all[-2 * self.look_per_epochs:-1 * self.look_per_epochs].min() > spot_convergence:
            #         self.period_thi += 1
            #         if self.period_thi == len(self.parameters_d):
            #             select_idx = loss_all.argmin()
            #             select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
            #             for i in range(len(self.parameters_all)):
            #                 self.parameters_all[i].data = select_data[i].data
            #                 self.parameters_all[i].requires_grad = False
            #             self.flag_use_post = True
            #             self.period_all = 'post'
            #             self.last_epochs = self.now_epochs
            #             return
            #         print('period:' + self.period_all + '_' + str(self.period_thi))
            #         select_idx = loss_all.argmin()
            #         select_data = torch.Tensor(self.data_all[self.last_epochs:][select_idx]).to(self.device)
            #         for i in range(len(self.parameters_all)):
            #             self.parameters_all[i].data = select_data[i].data
            #         d_now = self.parameters_d[self.period_thi]
            #         d_now.data = torch.Tensor([round(d_now.data.item(), round_num)]).to(self.device)[0]
            #         self.frozen_flag[self.period_thi + len(self.parameters_c)] = True
            #         self.last_epochs = self.now_epochs
            elif self.period_all == 'post' and (self.now_epochs - self.last_epochs) / self.look_per_epochs >= 2:
                loss_all = torch.Tensor(self.loss_all).to(self.device)[self.last_epochs:]
                if loss_all[-self.look_per_epochs:].min() - loss_all[-2 * self.look_per_epochs:-1 * self.look_per_epochs].min() > spot_convergence / 5:
                    self.final_loss = loss_all.min()
                    self.stop_flag = True
                    return

    def step_only_spot(self):
        device = self.loss_constraint.device
        (self.loss_rms + self.loss_constraint * self.weight_loss_constraint).backward()
        # Adam方法进行梯度更新
        self.datas = []
        self.grads = []
        for i, x in enumerate(self.parameters_all):
            self.datas.append(x.item())
            self.grads.append(x.grad.item())
        self.data_all.append(self.datas)
        self.loss_all.append((self.loss_rms + self.loss_constraint * self.weight_loss_constraint).item())
        self.adam_step()

    def step(self, PSF, d_PSF):
        device = PSF[0].device
        if self.loss_constraint > 0:
            (self.loss_constraint * self.weight_loss_constraint).backward()
        for i, x in enumerate(self.parameters_all):
            for psf_order in range(len(PSF)):
                x.grad = x.grad + torch.sum(PSF[psf_order].grad / self.train_rate * d_PSF[psf_order][:, :, i, :, :])  # 计算梯度

        # Adam方法进行梯度更新
        self.datas = []
        self.grads = []
        for i, x in enumerate(self.parameters_all):
            self.datas.append(x.item())
            self.grads.append(x.grad.item())
        self.data_all.append(self.datas)
        self.adam_step()

    def train_end2end(self, clear_img_s, blur_img_s, count):
        post_iter = self.post_iter
        # time1 = time.time()
        gt_image, cond_image = self.set_input_end2end(clear_img_s, blur_img_s, phase='train')
        # time2 = time.time()
        if self.flag_use_post:
            # 节省内存，增加时间
            # if self.net_name == 'SwinIR':
            #     i = np.random.randint(0, len(cond_image), 1)[0]
            #     out_img = self.model.netG(cond_image[i].unsqueeze(0))
            #     loss_post = F.mse_loss(gt_image[i].unsqueeze(0), out_img)
            # else:
            out_img = self.model.netG(cond_image)
        else:
            # if self.net_name == 'SwinIR':
            #     f_shuffle = torch.nn.PixelShuffle(4)
            #     loss_post = F.mse_loss(gt_image, f_shuffle(cond_image))
            # else:
            out_img = cond_image

        loss_pixel = F.mse_loss(gt_image, out_img)
        loss_perc = self.calc_lpips.calc(gt_image, out_img).mean()
        loss_simi = torch.zeros_like(loss_pixel)
        edof_num = len(clear_img_s)
        per_size = len(gt_image) // edof_num
        for ii in range(edof_num):
            if ii != edof_num//2:
                loss_simi = loss_simi + F.mse_loss(out_img[ii::(per_size-1)], out_img[edof_num//2::(per_size-1)])
        loss_post = loss_pixel + 0.01 * loss_perc + loss_simi * 0.1

        (self.weight_loss_network * loss_post).backward()
        # time3 = time.time()
        # print(time2 - time1)
        # print(time3-time2)
        if self.flag_use_post:
            if post_iter % 10 == 0:
                self.writer.add_scalar("loss_post", loss_post.item(), post_iter)
            if self.model.ema_scheduler is not None:
                if post_iter > self.model.ema_scheduler['ema_start'] and post_iter % self.model.ema_scheduler['ema_iter'] == 0:
                    self.model.EMA.update_model_average(self.model.netG_EMA, self.model.netG)
            # if (post_iter % self.test_iter == 0) and (post_iter > 0):
            #     print('test...')
            #     self.test_end2end(clear_img_s, blur_img_s)
        self.post_iter += 1
        return loss_post

    @torch.no_grad()
    def test_end2end(self, clear_img_s, blur_img_s):
        gt_image, blur_image = self.set_input_end2end(clear_img_s, blur_img_s, phase='test')
        n = blur_image.size(0)
        for i in range(n):
            name = 'iter{num}_{x}'.format(num=self.post_iter, x=i)
            save_path = self.model.save_root + name
            output = self.model.netG_EMA(blur_image[i].unsqueeze(0))
            utils.save_image(output, save_path + 'deblur.png')
            # if self.net_name == 'SwinIR':
            #     f_shuffle = torch.nn.PixelShuffle(4)
            #     utils.save_image(f_shuffle(blur_image[i].unsqueeze(0)), save_path + 'blur.png')
            # else:
            utils.save_image(blur_image[i], save_path + 'blur.png')
            utils.save_image(gt_image[i], save_path + 'gt.png')
            self.model.save_everything(model_root=self.model_root)

    def set_input_end2end(self, clear_img_list, blur_img_list, phase='train'):
        train_blur_img_s = None
        train_clear_img_s = None
        device = clear_img_list[0].device
        edof_num = len(clear_img_list)
        n, c, h, w = clear_img_list[0].shape
        clear_img_tensor = torch.zeros(edof_num, n, c, h, w).to(device)
        blur_img_tensor = torch.zeros_like(clear_img_tensor)
        for i in range(edof_num):
            clear_img_tensor[i] = clear_img_list[i]
            blur_img_tensor[i] = blur_img_list[i]
        if phase == 'train':
            b_size = self.post_b_size
            # if self.net_name == 'SwinIR':
            #     pixel_unshuffle = torch.nn.PixelUnshuffle(4)
            #     train_clear_img_s = clear_img_tensor.reshape(edof_num*n, c, h, w)
            #     train_blur_img_s = pixel_unshuffle(blur_img_tensor.reshape(edof_num*n, c, h, w))
            # else:
            img_size = self.post_img_size
            train_blur_img_s = torch.zeros([b_size * edof_num, 3, img_size, img_size]).to(device)
            train_clear_img_s = torch.zeros([b_size * edof_num, 3, img_size, img_size]).to(device)
            for j in range(b_size):
                index = np.random.randint(0, n)
                image_clear = clear_img_tensor[:, index]
                image_blur = blur_img_tensor[:, index]
                h_randint = np.random.randint(0, h - img_size)
                w_randint = np.random.randint(0, w - img_size)
                train_blur_img_s[j*edof_num: (j+1)*edof_num] = image_blur[:, :, h_randint:h_randint + img_size, w_randint:w_randint + img_size]
                train_clear_img_s[j*edof_num: (j+1)*edof_num] = image_clear[:, :, h_randint:h_randint + img_size, w_randint:w_randint + img_size]
        if phase == 'test':
            train_clear_img_s = clear_img_tensor.reshape(edof_num*n, c, h, w)
            # if self.net_name == 'SwinIR':
            #     pixel_unshuffle = torch.nn.PixelUnshuffle(4)
            #     train_blur_img_s = pixel_unshuffle(blur_img_tensor.reshape(edof_num * n, c, h, w))
            # else:
            train_blur_img_s = blur_img_tensor.reshape(edof_num*n, c, h, w)
        return train_clear_img_s, train_blur_img_s

    @torch.no_grad()
    def val_end2end(self, clear_img_s, blur_img_s, save=False, index=None):
        device = clear_img_s[0].device
        gt_image, blur_image = self.set_input_end2end(clear_img_s, blur_img_s, phase='test')
        n = blur_image.size(0)
        PSNR = []
        SSIM = []
        LPIPS = []
        LOSS = []
        out_put_all = []
        for i in range(n):
            name = 'val_iter{num}_count{y}_{x}'.format(num=self.post_iter, x=i, y=index)
            save_path = self.model.save_root + name
            if self.flag_use_post:
                output = self.model.netG_EMA(blur_image[i].unsqueeze(0))
            else:
                # if self.net_name == 'SwinIR':
                #     f_shuffle = torch.nn.PixelShuffle(4)
                #     output = f_shuffle(blur_image[i].unsqueeze(0))
                # else:
                output = blur_image[i].unsqueeze(0)
            if save:
                if self.flag_use_post:
                    utils.save_image(output, save_path + 'deblur.png')
                # if self.net_name == 'SwinIR':
                #     f_shuffle = torch.nn.PixelShuffle(4)
                #     utils.save_image(f_shuffle(blur_image[i].unsqueeze(0)), save_path + 'blur.png')
                # else:
                utils.save_image(blur_image[i], save_path + 'blur.png')
                utils.save_image(gt_image[i], save_path + 'gt.png')

            psnr_single = calculate_psnr(output, gt_image[i].unsqueeze(0))
            ssim_single = calculate_ssim(output, gt_image[i].unsqueeze(0))
            LPIPS_single = self.calc_lpips.calc(output, gt_image[i].unsqueeze(0)).mean().item()
            loss_single = 0.1 ** (psnr_single / 10) + LPIPS_single * 0.01
            LOSS.append(loss_single)
            PSNR.append(psnr_single)
            SSIM.append(ssim_single)
            LPIPS.append(LPIPS_single)
            out_put_all.append(output)
        loss_simi = 0
        for i in range(n):
            if i != n//2:
                loss_simi = loss_simi + F.mse_loss(out_put_all[i], out_put_all[n//2])
        loss_final = sum(LOSS)/n + loss_simi.item() * 0.1

        return sum(PSNR)/n, sum(SSIM)/n, sum(LPIPS)/n, loss_final

    @torch.no_grad()
    def val_final(self, clear_img_s, blur_img_s, save=False, index=None):
        device = clear_img_s[0].device
        gt_image, blur_image = self.set_input_end2end(clear_img_s, blur_img_s, phase='test')
        n = blur_image.size(0)
        PSNR = []
        SSIM = []
        LPIPS = []
        LOSS = []
        out_put_all = []
        for i in range(n):
            name = 'val_iter{num}_count{y}_{x}'.format(num=self.post_iter, x=i, y=index)
            save_path = self.model.save_root + name
            if self.flag_use_post:
                output = self.model.netG_EMA(blur_image[i].unsqueeze(0))
            else:
                # if self.net_name == 'SwinIR':
                #     f_shuffle = torch.nn.PixelShuffle(4)
                #     output = f_shuffle(blur_image[i].unsqueeze(0))
                # else:
                output = blur_image[i].unsqueeze(0)
            if save:
                if self.flag_use_post:
                    utils.save_image(output, save_path + 'deblur.png')
                # if self.net_name == 'SwinIR':
                #     f_shuffle = torch.nn.PixelShuffle(4)
                #     utils.save_image(f_shuffle(blur_image[i].unsqueeze(0)), save_path + 'blur.png')
                # else:
                utils.save_image(blur_image[i], save_path + 'blur.png')
                utils.save_image(gt_image[i], save_path + 'gt.png')

            psnr_single = calculate_psnr(output, gt_image[i].unsqueeze(0))
            ssim_single = calculate_ssim(output, gt_image[i].unsqueeze(0))
            LPIPS_single = self.calc_lpips.calc(output, gt_image[i].unsqueeze(0)).mean().item()
            loss_single = 0.1 ** (psnr_single / 10) + LPIPS_single * 0.01
            LOSS.append(loss_single)
            PSNR.append(psnr_single)
            SSIM.append(ssim_single)
            LPIPS.append(LPIPS_single)
            out_put_all.append(output)
        loss_simi = 0
        for i in range(n):
            if i != n // 2:
                loss_simi = loss_simi + F.mse_loss(out_put_all[i], out_put_all[n // 2])
        loss_final = sum(LOSS) / n + loss_simi.item() * 0.1

        return PSNR, SSIM, LPIPS, loss_final

    def adam_init(self):
        self.device = self.param_groups[0]['params'].device
        self.length = len(self.param_groups)
        self.m = torch.zeros(self.length).to(self.device)
        self.v = torch.zeros(self.length).to(self.device)
        self.t = torch.zeros(self.length).to(self.device)
        self.frozen_flag = torch.zeros(self.length).to(self.device).bool()

    def adam_step(self):
        for i in range(self.length):
            self.t[i] += 1
            parameters = self.param_groups[i]
            grad = parameters['params'].grad
            beta1 = parameters['betas'][0]
            beta2 = parameters['betas'][1]
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * grad
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (grad ** 2)

        for i in range(self.length):
            parameters = self.param_groups[i]
            if 0 <= i - len(self.parameters_c) < len(self.parameters_d):
                if i == len(self.parameters_c):
                    x = parameters['params']
                else:
                    x = self.param_groups[i]['params'] - self.param_groups[i - 1]['params']
            else:
                x = parameters['params']

            constraint = self.constraint_all[i]
            if (x <= constraint[0] and self.m[i] >= 0) or (x >= constraint[1] and self.m[i] <= 0) or (
                    self.now_epochs >= self.pre_epochs) or (self.frozen_flag[i]):
                self.param_groups[i]['lr'] = 0
                if 0 < i - len(self.parameters_c) < len(self.parameters_d) and self.now_epochs <= self.pre_epochs:
                    if (x <= constraint[0] and self.m[i - 1] <= 0) or (x >= constraint[1] and self.m[i - 1] >= 0):
                        self.param_groups[i - 1]['lr'] = 0
            else:
                # self.param_groups[i]['lr'] = 2 * self.lr_all[i] * self.adam_schedule(self.now_epochs)
                self.param_groups[i]['lr'] = self.lr_all[i]

        for i in range(self.length):
            parameters = self.param_groups[i]
            beta1 = parameters['betas'][0]
            beta2 = parameters['betas'][1]
            m_t_hat = self.m[i] / (1 - beta1 ** self.t[i])
            v_t_hat = self.v[i] / (1 - beta2 ** self.t[i])
            lr = self.param_groups[i]['lr']
            parameters['params'].data = parameters['params'].data - lr * m_t_hat / (1e-8 + v_t_hat ** 0.5)

    def adam_zero_grad(self):
        for i in range(self.length):
            parameters = self.param_groups[i]
            if parameters['params'].grad is not None:
                parameters['params'].grad.data = torch.Tensor([0])[0].to(self.device)

    def adam_schedule(self, iteration, max_pos=0.3):
        max_iter = self.pre_epochs * max_pos
        if iteration < max_iter:
            res = (iteration / max_iter) + 1e-5
        else:
            res = 1 - (iteration - max_iter) / (self.pre_epochs - max_iter)
        return res



