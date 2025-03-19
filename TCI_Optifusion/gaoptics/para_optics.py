import os
import random
import torch
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, resize
from torchvision import utils
import numpy as np
from .order1_func import Order1
from .constraint_func import Constraint
from .simulation_func import Simulation
from .diff_func import Diff

from .gy_basics import *
import pandas as pd
import torch.optim as optim
import torch.nn as nn


# 绘制透镜界面：316~568行
# 光线采样：610~875行
class Path:
    def __init__(self):
        self.optics_read_path = None
        self.clear_root = None
        self.blur_root = None
        self.input_root = None
        self.demo_root = None
        self.save_root = None


class Basics:
    def __init__(self):
        self.device = None
        self.aper_index = None
        self.fov = None
        self.fnum = None
        self.acc_fit = None
        self.acc_range = None
        self.z_object = None
        self.z_object_edof = None
        self.img_r = None
        self.structure = None
        self.fix_vari = None
        self.max_r = None

        self.save_root = None
        self.demo_root = None
        self.init_read_root = None
        self.material_root = None

        self.CVVA = None  # 曲率值
        self.CTVAG = None  # 玻璃中心厚度
        self.CTVAA = None  # 空气中心间距
        self.COVA = None  # 二次项（圆锥系数）
        self.ASPH = None  # 非球面系数
        self.INDX = None  # 材料折射率
        self.MVAB = None  # 材料阿贝数

        self.ETVAG = None  # 玻璃边缘厚度
        self.ETVAA = None  # 空气边缘间距
        self.BFFL = None  # 后工作距
        self.EFFL = None  # 有效焦距
        self.TTVA = None  # 系统总长，也即是从第一面到像面的距离
        self.RIMH = None  # 边缘视场像高
        self.DISG = None  # 畸变

        self.fov_sample_num = None  # 视场采样数量
        self.enp_sample_num = None  # 入瞳光线采样数量
        self.max_pop = None
        self.max_gener_num = None
        self.sphere_gener_num = None
        self.asphere_gener_num = None
        self.wave_sample = None

        self.use_aspheric = None
        self.use_doe = None

        self.acc_fit_all = []
        self.distort_all = []
        self.fit_all = []

        self.z_object = None
        self.z_sensor = None

        self.surfaces = None
        self.materials = None

        self.min_F_num = None
        self.F_num_all = None
        self.MAX_R = None
        self.defocus = None
        self.aper_R = None

        self.surface_num = None
        self.piece_num = None
        self.aspheric_order = []
        self.aspheric_list = []
        self.conic_list = []
        self.sphere_list = []
        self.material_list = []
        self.d_list = []
        self.aper_list = []

        self.index_c = None
        self.index_d = None
        self.index_k = None
        self.index_n = None
        self.index_V = None
        self.index_asp = None
        self.index_defocus = None

        self.len_inputs = None
        self.phi_inf = None
        self.valid_idx = None
        self.valid_idx_now = None

        self.gener_num = 0
        self.draw_num = 0
        self.exp_num = 0
        self.exp_z_sensor = []

        self.acc_data = None

        self.val_flag = False
        self.final_select_flag = True


class LensPopulation:
    """
    定义透镜组，分为基本数据和高阶数据
    除波长外单位一般均为[mm]
    位置是相对透镜组第一个透镜面（不包括孔径光阑）的距离
    """

    def __init__(self):
        self.basics = Basics()
        self.simulation = Simulation()
        self.constraint = Constraint()
        self.order1 = Order1()
        self.diff = Diff()
        self.path = Path()

    def init_pop(self, flag, sphere_structure=None):
        device = self.basics.device

        self.order1.lr_rate = []
        wavelength_all = self.basics.wave_sample
        self.order1.wave_data = {
            "all": wavelength_all['R'][0] + wavelength_all['G'][0] + wavelength_all['B'][0],
            "all_weight": wavelength_all['R'][1] + wavelength_all['G'][1] + wavelength_all['B'][1],
            "main_RGB": [np.median(wavelength_all['R'][0]), np.median(wavelength_all['G'][0]),
                         np.median(wavelength_all['B'][0])],
            "main": float(np.median(wavelength_all['G'][0]))
        }
        if self.basics.material_root is not None:
            self.order1.material_lab = Material_lab(self.basics.material_root, self.order1.wave_data['all'])
            mater_n = torch.Tensor(self.order1.material_lab.nd).to(device)[1:]
            mater_v = torch.Tensor(self.order1.material_lab.vd).to(device)[1:]
            self.basics.INDX = [mater_n.min().item(), mater_n.max().item()]
            self.basics.MVAB = [mater_v.min().item(), mater_v.max().item()]
        self.order1.c_list = []
        self.order1.d_list = []
        self.order1.n_list = []
        self.order1.v_list = []
        self.order1.now_structures = []
        for i in range(len(self.basics.structure)):
            now_structure = self.basics.structure[i]
            if now_structure[0] == 'S':
                self.order1.now_structures.append('S0' + now_structure[2])
        self.order1.surface_num = len(self.order1.now_structures)
        self.order1.len_parameters = 0

        self.order1.phi_inf = []
        for i in range(self.order1.surface_num):
            now_structure = self.order1.now_structures[i]
            self.order1.c_list.append(self.order1.len_parameters)
            self.order1.lr_rate.append(1)
            self.order1.len_parameters += 1
            self.order1.phi_inf.append([self.basics.CVVA[0], self.basics.CVVA[1] - self.basics.CVVA[0]])
            if now_structure[2] == 'G':
                self.order1.d_list.append(self.order1.len_parameters)
                self.order1.n_list.append(self.order1.len_parameters + 1)
                self.order1.v_list.append(self.order1.len_parameters + 2)
                self.order1.lr_rate.append(1)
                self.order1.lr_rate.append(1)
                self.order1.lr_rate.append(1)

                self.order1.len_parameters += 3
                self.order1.phi_inf.append([self.basics.CTVAG[0], self.basics.CTVAG[1] - self.basics.CTVAG[0]])
                self.order1.phi_inf.append([self.basics.INDX[0], self.basics.INDX[1] - self.basics.INDX[0]])
                self.order1.phi_inf.append([self.basics.MVAB[0], self.basics.MVAB[1] - self.basics.MVAB[0]])

            elif now_structure[2] == 'A':
                if i == self.order1.surface_num - 1:
                    self.order1.phi_inf.append([self.basics.BFFL[0], self.basics.BFFL[1] - self.basics.BFFL[0]])
                else:
                    self.order1.d_list.append(self.order1.len_parameters)
                    self.order1.phi_inf.append([self.basics.CTVAA[0], self.basics.CTVAA[1] - self.basics.CTVAA[0]])
                self.order1.lr_rate.append(1)
                self.order1.len_parameters += 1

        self.order1.vari_flag = torch.ones(self.order1.len_parameters).bool().to(device)
        self.order1.fix_data = torch.zeros(self.order1.len_parameters).to(device)
        res = torch.rand(self.basics.max_pop, self.order1.len_parameters).to(device).double()

        if self.basics.material_root is not None:
            mater_index = torch.randint(0, len(mater_n), (len(res), len(self.order1.n_list))).to(device)
            res[:, torch.Tensor(self.order1.n_list).long()] = ((mater_n[mater_index] - self.basics.INDX[0]) / (
                    self.basics.INDX[1] - self.basics.INDX[0])).double()
            res[:, torch.Tensor(self.order1.v_list).long()] = ((mater_v[mater_index] - self.basics.MVAB[0]) / (
                    self.basics.MVAB[1] - self.basics.MVAB[0])).double()

        for i in range(len(self.basics.fix_vari)):
            item = self.basics.fix_vari[i]
            order = item[0]
            name = item[1]
            if name == 'c':
                res[:, self.order1.c_list[order]] = (item[2] - self.basics.CVVA[0]) / (
                        self.basics.CVVA[1] - self.basics.CVVA[0])
                self.order1.vari_flag[self.order1.c_list[order]] = False
                self.order1.fix_data[self.order1.c_list[order]] = (item[2] - self.basics.CVVA[0]) / (
                        self.basics.CVVA[1] - self.basics.CVVA[0])
            if name == 'd':
                if self.order1.now_structures[order][2] == 'G':
                    res[:, self.order1.d_list[order]] = (item[2] - self.basics.CTVAG[0]) / (
                            self.basics.CTVAG[1] - self.basics.CTVAG[0])
                    self.order1.fix_data[self.order1.d_list[order]] = (item[2] - self.basics.CTVAG[0]) / (
                            self.basics.CTVAG[1] - self.basics.CTVAG[0])
                if self.order1.now_structures[order][2] == 'A':
                    res[:, self.order1.d_list[order]] = (item[2] - self.basics.CTVAA[0]) / (
                            self.basics.CTVAA[1] - self.basics.CTVAA[0])
                    self.order1.fix_data[self.order1.d_list[order]] = (item[2] - self.basics.CTVAA[0]) / (
                            self.basics.CTVAA[1] - self.basics.CTVAA[0])

                self.order1.vari_flag[self.order1.d_list[order]] = False
            if name == 'n':
                res[:, self.order1.n_list[order]] = (item[2] - self.basics.INDX[0]) / (
                        self.basics.INDX[1] - self.basics.INDX[0])
                self.order1.vari_flag[self.order1.n_list[order]] = False
                self.order1.fix_data[self.order1.n_list[order]] = (item[2] - self.basics.INDX[0]) / (
                        self.basics.INDX[1] - self.basics.INDX[0])
            if name == 'v':
                res[:, self.order1.v_list[order]] = (item[2] - self.basics.MVAB[0]) / (
                        self.basics.MVAB[1] - self.basics.MVAB[0])
                self.order1.vari_flag[self.order1.v_list[order]] = False
                self.order1.fix_data[self.order1.v_list[order]] = (item[2] - self.basics.MVAB[0]) / (
                        self.basics.MVAB[1] - self.basics.MVAB[0])

        if self.basics.TTVA is not None:
            fix_d = list(torch.where(~self.order1.vari_flag)[0])
            list_vari_d = self.order1.d_list.copy()
            ttl = self.basics.TTVA[1] - self.basics.BFFL[0]
            for i in range(len(fix_d)):
                for j in range(len(list_vari_d)):
                    if fix_d[i] == list_vari_d[j]:
                        ttl = ttl - (self.order1.fix_data[list_vari_d[j]] * self.order1.phi_inf[list_vari_d[j]][1] +
                                     self.order1.phi_inf[list_vari_d[j]][0])
                        list_vari_d.remove(list_vari_d[j])
                        break
            ttl_rate = torch.rand(res.size(0), len(list_vari_d)).to(device)
            ttl_rate = (ttl_rate.transpose(0, 1) / ttl_rate.sum(dim=1)).transpose(0, 1)
            ttl_all = ttl * ttl_rate
            d_range = torch.Tensor(self.order1.phi_inf).to(device)[list_vari_d]
            d_init = torch.clamp((ttl_all - d_range[:, 0]) / d_range[:, 1], 0.01, 0.6)
            res[:, list_vari_d] = d_init.double()

            # if self.basics.EFFL is not None:
            #     fix_c = list(torch.where(~self.order1.vari_flag)[0])
            #     list_vari_c = self.order1.c_list.copy()
            #     focal_all = 2 / (self.basics.EFFL[1] + self.basics.EFFL[0])
            #     for i in range(len(fix_c)):
            #         for j in range(len(list_vari_c)):
            #             if fix_c[i] == list_vari_c[j]:
            #                 # n2 = torch.ones(self.basics.max_pop).to(device)
            #                 # n1 = torch.ones(self.basics.max_pop).to(device)
            #                 # if diff_c_list[j] == 4:
            #                 #     n2 = res[:, list_vari_c[j]+2] * (self.basics.INDX[1]-self.basics.INDX[0]) + self.basics.INDX[0]
            #                 # if j > 0 and diff_c_list[j-1] == 4:
            #                 #     n1 = res[:, list_vari_c[j] + 2] * (self.basics.INDX[1] - self.basics.INDX[0]) + \
            #                 #          self.basics.INDX[0]
            #                 list_vari_c.remove(list_vari_c[j])
            #                 break
            #     diff_c_list = list(torch.Tensor(list_vari_c).to(device).diff())
            #     diff_c_list.append(2)
            #     focal_rate = torch.rand(res.size(0), len(list_vari_c)).to(device) - 0.3
            #     focal_rate = (focal_rate.transpose(0, 1) / focal_rate.sum(dim=1)).transpose(0, 1)
            #     focal_each = focal_all * focal_rate
            #     c_range = torch.Tensor(self.order1.phi_inf).to(device)[list_vari_c]
            #     for j in range(len(list_vari_c)):
            #         n2 = torch.ones(self.basics.max_pop).to(device)
            #         n1 = torch.ones(self.basics.max_pop).to(device)
            #         if diff_c_list[j] == 4:
            #             n2 = res[:, list_vari_c[j] + 2] * (self.basics.INDX[1] - self.basics.INDX[0]) + \
            #                  self.basics.INDX[0]
            #         if j > 0 and diff_c_list[j - 1] == 4:
            #             n1 = res[:, list_vari_c[j] - 2] * (self.basics.INDX[1] - self.basics.INDX[0]) + \
            #                  self.basics.INDX[0]
            #         focal_now = focal_each[:, j]
            #         c_range_now = c_range[j]
            #         res[:, list_vari_c[j]] = ((focal_now / (n2 - n1)) - c_range_now[0]) / c_range_now[1]

        if self.basics.init_read_root is not None:
            data = pd.read_excel(self.basics.init_read_root).values
            i = 0
            num_optics = 0
            while i < len(data):
                if isinstance(data[i, 0], str) and data[i, 0][:3] == 'num':
                    para_order = 0
                    i = i + 3
                    for k in range(self.order1.surface_num):
                        now_structure = self.basics.structure[k]
                        now_data = data[i + k]
                        if now_structure[0] == 'S':
                            order = int(now_structure[1])
                            for j in range(order + 1):
                                if j == 0:
                                    if now_data[3] == '无限':
                                        now_data[3] = 1e10
                                    res[num_optics, para_order] = (1 / now_data[3] - self.order1.phi_inf[para_order][0]) / self.order1.phi_inf[para_order][1]
                                    para_order += 1
                        if now_structure[2] == 'G':
                            res[num_optics, para_order] = (now_data[4] - self.order1.phi_inf[para_order][
                                0]) / self.order1.phi_inf[para_order][1]
                            para_order += 1
                            res[num_optics, para_order] = (float(now_data[5].split(';')[0]) -
                                                           self.order1.phi_inf[para_order][
                                                               0]) / self.order1.phi_inf[para_order][1]
                            para_order += 1

                            res[num_optics, para_order] = (float(now_data[5].split(';')[1]) -
                                                           self.order1.phi_inf[para_order][
                                                               0]) / self.order1.phi_inf[para_order][1]
                            para_order += 1

                        elif now_structure[2] == 'A':
                            res[num_optics, para_order] = (now_data[4] - self.order1.phi_inf[para_order][
                                0]) / self.order1.phi_inf[para_order][1]
                            para_order += 1

                    num_optics += 1
                    i += self.order1.surface_num + 1

                else:
                    i += 1
        self.order1.init_population = res

        # for i in range(self.order1.surface_num):
        #     now_structure = self.order1.now_structures[i]
        #     if now_structure[0] == 'S':
        #         order = int(now_structure[1])
        #         self.order1.len_parameters += order + 1
        #         for j in range(order + 1):
        #             if j == 0:
        #                 self.order1.phi_inf.append([self.basics.CVVA[0], self.basics.CVVA[1] - self.basics.CVVA[0]])
        #             elif j == 1:
        #                 self.order1.phi_inf.append([self.basics.COVA[0], self.basics.COVA[1] - self.basics.COVA[0]])
        #             else:
        #                 self.order1.phi_inf.append([self.basics.ASPH[0], self.basics.ASPH[1] - self.basics.ASPH[0]])
        #     if now_structure[2] == 'G':
        #         self.order1.len_parameters += 3
        #         self.order1.phi_inf.append([self.basics.CTVAG[0], self.basics.CTVAG[1] - self.basics.CTVAG[0]])
        #         self.order1.phi_inf.append([self.basics.INDX[0], self.basics.INDX[1] - self.basics.INDX[0]])
        #         self.order1.phi_inf.append([self.basics.MVAB[0], self.basics.MVAB[1] - self.basics.MVAB[0]])
        #
        #     elif now_structure[2] == 'A':
        #         self.order1.len_parameters += 1
        #         self.order1.phi_inf.append([self.basics.CTVAA[0], self.basics.CTVAA[1] - self.basics.CTVAA[0]])
        #
        # self.order1.vari_flag = torch.ones(self.order1.len_parameters).bool().to(device)
        # res = torch.rand(self.basics.max_pop, self.order1.len_parameters).to(device).double()
        # self.order1.init_population = res

    def set_basics(self, sol):
        device = sol.device
        self.order1.phi_inf = torch.Tensor(self.order1.phi_inf).to(device)
        sol = self.order1.phi_inf[:, 0] + self.order1.phi_inf[:, 1] * sol
        sol = sol.transpose(0, 1).float()
        r_init = torch.ones(sol.size(1)).to(device) * self.basics.max_r

        para_order = 0
        self.order1.surfaces = []
        self.order1.materials = [Material(name='air', pop=sol, MATERIAL_TABLE=self.order1.material_lab)]
        now_d = torch.zeros(sol.size(1)).to(device)
        sol_d = torch.zeros(sol.size(1)).to(device)
        for i in range(self.order1.surface_num):
            now_structure = self.order1.now_structures[i]
            sol_c = sol[para_order]
            para_order += 1
            if i == 0:
                now_d = torch.zeros_like(sol[para_order])
            else:
                now_d = now_d + sol_d
            sol_d = sol[para_order]
            if i == self.order1.surface_num - 1:
                self.order1.z_sensor = now_d + sol_d
            para_order += 1
            if now_structure[2] == 'G':
                sol_n = sol[para_order]
                para_order += 1
                sol_V = sol[para_order]
                para_order += 1
                mater_item = Material(data=[sol_n, sol_V], MATERIAL_TABLE=self.order1.material_lab, use_real_glass=self.order1.use_real_glass)
            else:
                mater_item = Material(name='air', pop=sol, MATERIAL_TABLE=self.order1.material_lab, use_real_glass=self.order1.use_real_glass)
            surface = Aspheric(sol_c, now_d, r_init)
            self.order1.surfaces.append(surface)
            self.order1.materials.append(mater_item)

        # if self.order1.stage == 'asphere':
        #     for i in range(self.order1.surface_num):
        #         surface = self.order1.surfaces[i]
        #         now_structure = self.order1.now_structures[i]
        #         order = int(now_structure[1])
        #         if order >= 1:
        #             sol_k = sol[para_order]
        #             surface.k = sol_k
        #             para_order += 1
        #         if order >= 2:
        #             ai_item = sol[para_order: para_order + order - 1]
        #             para_order += order - 1
        #             surface.ai = ai_item
        self.order1.valid_idx = torch.ones(sol.size(1)).bool().to(device)
        #
        # device = sol.device
        # self.order1.phi_inf = torch.Tensor(self.order1.phi_inf).to(device)
        # sol = self.order1.phi_inf[:, 0] + self.order1.phi_inf[:, 1] * sol
        # sol = sol.transpose(0, 1).float()
        # r_init = torch.ones(sol.size(1)).to(device) * self.basics.max_r
        # para_order = 0
        # self.order1.surfaces = []
        # self.order1.materials = [Material(name='air', pop=sol, MATERIAL_TABLE=self.order1.material_lab)]
        # now_d = torch.zeros(sol.size(1)).to(device)
        # sol_d = torch.zeros(sol.size(1)).to(device)
        # for i in range(self.order1.surface_num):
        #     now_structure = self.basics.structure[i]
        #     sol_k = None
        #     ai_item = None
        #     if now_structure[0] == 'S':
        #         order = int(now_structure[1])
        #         sol_c = sol[para_order]
        #         para_order += 1
        #         if order >= 1:
        #             sol_k = sol[para_order]
        #             para_order += 1
        #         if order >= 2:
        #             ai_item = sol[para_order: para_order + order - 1]
        #             para_order += order - 1
        #
        #         if i == 0:
        #             now_d = torch.zeros_like(sol[para_order])
        #         else:
        #             now_d = now_d + sol_d
        #         sol_d = sol[para_order]
        #         if i == self.order1.surface_num - 1:
        #             self.order1.z_sensor = now_d + sol_d
        #         para_order += 1
        #
        #         if now_structure[2] == 'G':
        #             sol_n = sol[para_order]
        #             para_order += 1
        #             sol_V = sol[para_order]
        #             para_order += 1
        #             mater_item = Material(data=[sol_n, sol_V])
        #
        #         else:
        #             mater_item = Material(name='air', pop=sol, MATERIAL_TABLE=self.order1.material_lab)
        #         surface = Aspheric(sol_c, now_d, r_init, k=sol_k, ai=ai_item)
        #
        #         self.order1.surfaces.append(surface)
        #         self.order1.materials.append(mater_item)
        #
        # self.order1.valid_idx = torch.ones(sol.size(1)).bool().to(device)

    def find_asphere(self, sol, mutated_surface=2, iter_num=5000, beta1=0.9, beta2=0.9):
        item = self.basics.max_pop
        self.basics.max_pop = len(sol)
        device = sol.device
        self.re_init()
        t1 = time.time()  # 运行时间计算
        self.set_basics(sol)
        self.calc_order1()
        idx_as = []
        for sur_num in range(self.order1.surface_num):
            surface = self.order1.surfaces[sur_num]
            if surface.ai is not None:
                # 自动计算一阶参数
                idx_as.append(sur_num)

        def cal_surface(para, phi, r):
            para = (phi[:, 0] + para.T * phi[:, 1]).T
            c = para[0]
            k = para[1]
            ai = para[2:]
            alpha = (1 + k) * c.square()
            rou = r.square()
            h = c * rou / (1 + torch.clamp(1 - alpha * rou, min=1e-8).sqrt())
            for ii in range(len(ai)):
                h = h + ai[ii] * (rou ** (ii + 2))
            return h

        def loss_suf(x, y):
            loss = (x - y).square().mean(dim=0)

            return loss

        def loss_simi(p1, p2):
            item_rms = (p1 - p2).square().sum(dim=0).sqrt() / len(p1)
            item_rms[item_rms > 0.1] = 1
            return 100 * (1 - item_rms)

        for i in range(mutated_surface):
            random_index = random.randint(0, len(idx_as) - 1)
            random_element = idx_as[random_index]
            surface = self.order1.surfaces[random_element]
            idx_c = 0
            final_c = 0
            final_k = 0
            for j in range(len(self.order1.now_structures)):
                if j == random_element:
                    final_c = idx_c
                structure = self.order1.now_structures[j]
                if structure[0] == 'S':
                    if structure[2] == 'G':
                        idx_c += 4
                    if structure[2] == 'A':
                        idx_c += 2

            for j in range(len(self.order1.now_structures)):
                if j == random_element:
                    final_k = idx_c
                structure = self.order1.now_structures[j]
                idx_c += int(structure[1])
            phi_now = torch.zeros(2 + len(surface.ai), 2).to(device)
            phi_now[0] = self.order1.phi_inf[final_c]
            phi_now[1] = self.order1.phi_inf[final_k]
            phi_now[2:] = self.order1.phi_inf[final_k + 1:final_k + 1 + len(surface.ai)]

            para_gt = ((torch.cat((surface.c.unsqueeze(0), surface.k.unsqueeze(0), surface.ai)).T - phi_now[:,
                                                                                                    0]) / phi_now[:,
                                                                                                          1]).T
            r_sample = surface.r.unsqueeze(0) * torch.linspace(0, 1, 50).to(device).unsqueeze(1)
            z_gt = cal_surface(para_gt, phi_now, r_sample)
            para_start = torch.rand(2 + len(surface.ai), len(surface.c)).to(device)

            para_real = (phi_now[:, 0] + para_start.T * phi_now[:, 1]).T
            k_max = 1 / (1 / (para_real[0].square() * surface.r.square()) - 1)
            para_real[1][para_real[1] - k_max > 0] = k_max[para_real[1] - k_max > 0]
            para_start = ((para_real.T - phi_now[:, 0]) / phi_now[:, 1]).T

            lr = 1e-3
            delta_x = 1e-4
            l, n = para_start.size()
            m = torch.zeros(l, n).to(device)
            v = torch.zeros(l, n).to(device)
            t = torch.zeros(l, n).to(device)
            print('find_asphere ...')
            # loss_start = loss_suf(z_gt, cal_surface(para_start, phi_now, r_sample)) + loss_simi(para_gt, para_start)
            for k in range(iter_num):
                t += 1
                grad = torch.zeros_like(para_start)
                for q in range(len(grad)):
                    item_para = para_start.clone()
                    item_para[q] += delta_x
                    loss1 = loss_suf(z_gt, cal_surface(item_para, phi_now, r_sample)) + loss_simi(para_gt, item_para)
                    item_para = para_start.clone()
                    item_para[q] -= delta_x
                    loss0 = loss_suf(z_gt, cal_surface(item_para, phi_now, r_sample)) + loss_simi(para_gt, item_para)
                    grad[q] = (loss1 - loss0) / (2 * delta_x)
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * (grad ** 2)
                m_t_hat = m / (1 - beta1 ** t)
                v_t_hat = v / (1 - beta2 ** t)
                delta_x_final = lr * m_t_hat / (1e-8 + v_t_hat ** 0.5)
                para_start = torch.clamp((para_start - delta_x_final), 0, 1)

                # loss_now = loss_suf(z_gt, cal_surface(para_start, phi_now, r_sample)) + 100 * loss_simi(para_gt, para_start)
                # print(loss_now.mean().item())
            sol[self.order1.valid_idx, final_c] = para_start[0]
            sol[self.order1.valid_idx, final_k: final_k + 1 + len(surface.ai)] = para_start[1:].T


            # valid_flag = torch.ones_like(c_now).bool()
            # while valid_flag.any():

        # t2 = time.time()
        # M = self.basics.enp_sample_num
        # views = self.order1.fov_samples
        # wave_len = self.order1.wave_data["all"][(self.order1.now_gener + 1) % (len(self.order1.wave_data["all"]))]
        # size_pop_now = self.order1.valid_idx.sum()
        # self.order1.valid_idx_now = torch.ones(self.order1.valid_idx.sum()).to(device).bool()
        # M2 = M ** 2
        # ray_o = torch.zeros(3, M2 * self.basics.fov_sample_num, size_pop_now).to(device)
        # ray_d = torch.zeros(3, M2 * self.basics.fov_sample_num, size_pop_now).to(device)
        # for i in range(self.basics.fov_sample_num):
        #     fov = views[i]
        #     ray = self.sample_ray_common(wave_len, view=fov, pos=self.order1.enp_xyz[:, i], M=M, mode='2D')
        #     ray_o[:, i * M2:(i + 1) * M2] = ray.o
        #     ray_d[:, i * M2:(i + 1) * M2] = ray.d
        # ray_all = Ray(ray_o, ray_d, wave_len)
        # valid_all, ray_final_all, oss_all = self._trace(ray_all, record=True)
        # ray_o = ray_final_all.o
        # ray_d = ray_final_all.d
        # for i in range(self.basics.fov_sample_num):
        #     ray_o[0, i * M2:(i + 1) * M2] = -self.basics.img_r * self.order1.fov_pos[i]
        # ray_d = -ray_d
        # ray_o[2] = self.order1.z_sensor
        # ray_back = Ray(ray_o, ray_d, wave_len)
        # valid_back, ray_back, oss_back = self._trace(ray_back, record=True, start_ind=sur_num)
        # ray_d_gt = oss_back[sur_num + 1] - oss_back[sur_num]
        # ray_o_gt = oss_back[sur_num + 1]
        #
        # ray_d_real = oss_all[sur_num + 1] - oss_all[sur_num]
        # ray_o_real = oss_all[sur_num]
        #
        # x1 = ray_o_gt[0]
        # y1 = ray_o_gt[2]
        # a1 = ray_d_gt[0]
        # b1 = ray_d_gt[2]
        # x2 = ray_o_real[0]
        # y2 = ray_o_real[2]
        # a2 = ray_d_real[0]
        # b2 = ray_d_real[2]
        #
        # x_gt = (a2*b1*x1 - b2*a1*x2 + a2*a1*(y2-y1)) / (a2*b1 - b2*a1)
        # z_gt = (a2*b1*y2 - b2*a1*y1 + b2*b1*(x1-x2)) / (a2*b1 - b2*a1)
        # z_gt = z_gt - surface.d
        # valid_now = valid_back & valid_all & (~torch.isnan(x_gt)) & (~torch.isnan(z_gt))
        # x_gt[~valid_now] = 0
        # z_gt[~valid_now] = 0
        # ray_exp = Ray(ray_o_real, ray_d_real, wave_len)
        # row = x_gt ** 2
        #
        # # 训练模型
        # epochs = 100
        # c_now = surface.c.clone()
        # k_now = surface.k.clone()
        # ai_now = surface.ai.clone()
        #
        # for t in range(epochs):
        #     alpha = (1 + k_now) * (c_now ** 2)
        #     z_real = (c_now * row) / (1 + torch.clamp(1-alpha*row, min=1e-8)**0.5)
        #     valid_all, ray_final_all, oss_all = self._trace(ray_exp, record=True, stop_ind=sur_num, start_ind=sur_num)
        #     z_d = ray_final_all.d
        #     cos_d = (z_d * ray_d_gt).sum(dim=0) / (ray_d_gt.square().sum(dim=0).sqrt() * z_d.square().sum(dim=0).sqrt())
        #     cos_d[~valid_now] = 1
        #     loss = 1000 * (1 - cos_d).square().sum(dim=0) + (z_real - z_gt).square().sum(dim=0)
        # # 实例化模型，设多项式的阶数为2

        self.basics.max_pop = item
        return sol

    def fitness_func(self, sol):
        """
        仿真准备工作，相关参数初始化计算
        """
        dev = sol.device
        self.re_init()
        t1 = time.time()  # 运行时间计算
        self.set_basics(sol)

        # 自动计算一阶参数
        self.calc_order1()

        t2 = time.time()
        fit_aberration = self.calc_fit_img()
        fit_constraint = self.calc_fit_constraint()
        fit_all = torch.ones(self.basics.max_pop).to(self.basics.device) * 1e3
        fit_all[self.order1.valid_idx] = fit_aberration + fit_constraint
        fit_all[torch.isnan(fit_all)] = 1e3
        self.order1.fit_all = fit_all

        self.order1.final_select_flag = True
        if self.order1.final_select_flag:
            final_divide_rate = self.basics.acc_range
            acc_idx = torch.where(fit_all < self.basics.acc_fit)[0]
            acc_data = sol[acc_idx]
            acc_fit_now = fit_all[acc_idx]

            fitness_sorted_idx = acc_fit_now.sort()[1]
            acc_fit_now_sorted = acc_fit_now.sort()[0]

            acc_idx_sorted = acc_idx[fitness_sorted_idx]
            acc_data_sorted = acc_data[fitness_sorted_idx]

            acc_idx2 = []
            self.order1.acc_data = []
            self.order1.acc_fit_all = []
            for i in range(len(acc_idx_sorted)):
                now_idx = acc_idx_sorted[i]
                acc_flag = False
                if len(self.order1.acc_data) == 0:
                    acc_flag = True
                else:
                    diff_fit = acc_fit_now_sorted[i] - torch.Tensor(self.order1.acc_fit_all).to(dev)
                    diff = (acc_data_sorted[i] - torch.Tensor(self.order1.acc_data).to(dev)).square().sum(
                        dim=-1).sqrt() / (sol.size(1) ** 0.5)
                    if diff.min(dim=0).values > final_divide_rate:
                        acc_flag = True
                    elif diff_fit[diff <= final_divide_rate].max() < -1e-3 and diff[
                        diff <= final_divide_rate].min() > final_divide_rate / 1000:
                        acc_flag = True

                if acc_flag:
                    self.order1.acc_data.append(acc_data_sorted[i].tolist())
                    self.order1.acc_fit_all.append(acc_fit_now_sorted[i])
                    acc_idx2.append(now_idx)

            acc_idx = torch.Tensor(acc_idx2).to(dev).long()
            # self.write_optics_data(acc_idx, save_path=self.basics.save_root)

            draw = True
            # 调试

            # 画光线追迹图
            if draw and len(acc_idx) > 0:
                sort_index = acc_idx
                self.draw_all(M=9, sort_index=sort_index)
        t3 = time.time()  # 运行时间计算
        # print('time_oder1:{time}'.format(time=t2 - t1))
        # print('time_rms:{time}'.format(time=t3 - t2))
        return fit_all

    def calc_order1(self, cal_time=False, flag_auto_flag=True):
        t1 = time.time()
        device = self.basics.device

        if flag_auto_flag:
            self.order1.aperture_max = self.calc_max_aper(wavelength=self.order1.wave_data["main"])
        # 此时也需要re_valid
        self.order1.aperture_max = torch.clamp(self.order1.aperture_max, min=self.basics.max_r / 100,
                                               max=self.basics.max_r)

        if self.basics.aper_R is not None:
            self.order1.aperture_max[self.order1.aperture_max < self.basics.aper_R[0]] = self.basics.aper_R[0]
            self.order1.aperture_max[self.order1.aperture_max > self.basics.aper_R[1]] = self.basics.aper_R[1]
        t2 = time.time()
        self.re_valid_func(self.order1.valid_idx)

        if flag_auto_flag:
            idx = self.order1.surfaces[self.basics.aper_index].r > self.order1.aperture_max
            self.order1.surfaces[self.basics.aper_index].r[idx] = self.order1.aperture_max[idx]

        self.order1.fov_samples, self.order1.fov_pos = self.calc_fov_samples(self.basics.fov, pos=[0, 0.707, 1])

        # 3、计算得到每个视场下主波长光线在第一面的入瞳坐标与相对照度（渐晕）因子, 尝试与resize_self合并
        self.order1.enp_xyz = self.calc_enp_all_fov(wavelength=self.order1.wave_data["main"])
        t3 = time.time()

        if cal_time:
            print('time_aper:{time}'.format(time=t2 - t1))
            print('time_enp:{time}'.format(time=t3 - t2))

    def calc_max_aper(self, wavelength=588.0, M=8):
        surfaces = self.order1.surfaces
        device = self.basics.device
        aperture_index = self.basics.aper_index
        size_pop = self.basics.max_pop

        time1 = time.time()
        for i in range(len(surfaces)):
            item = (1 + self.order1.surfaces[i].k) * (self.order1.surfaces[i].c ** 2)
            item[item <= 0] = 1e-6
            surfaces[i].r = (1 / item) ** 0.5
            surfaces[i].r[surfaces[i].r > self.basics.max_r] = self.basics.max_r
            sample_r = 5
            valid = torch.ones_like(item).bool()
            final_max_r = surfaces[i].r.clone()
            for j in range(sample_r):
                now_r = (j + 1) / sample_r * final_max_r.max()
                sag = surfaces[i].surface(now_r, 0.0)
                idx = (sag > self.basics.CTVAG[1] * 2) & valid
                final_max_r[idx] = now_r
                valid[idx] = False
                if (~valid).all():
                    break
            surfaces[i].r[surfaces[i].r > final_max_r] = final_max_r[surfaces[i].r > final_max_r]
        time2 = time.time()
        # print(time2 - time1)
        self.order1.effl = self.calc_effl(wavelength=wavelength)
        time3 = time.time()
        # print(time3 - time2)
        self.order1.enpd = (self.order1.effl / self.basics.fnum).abs()
        if self.basics.EFFL is not None:
            self.order1.enpd[(self.order1.effl > self.basics.EFFL[1])] = (self.basics.EFFL[1] / self.basics.fnum)
            self.order1.enpd[(self.order1.effl < self.basics.EFFL[0])] = (self.basics.EFFL[0] / self.basics.fnum)

        # 孔阑中心反向追迹，得到入瞳位置
        if aperture_index == 0:
            enpp = surfaces[0].d
            self.order1.enpp = enpp
            self.order1.valid_idx[self.order1.effl <= 0] = False
            return self.order1.enpd / 2
        else:
            o = torch.zeros(3, 1, size_pop).to(device)
            d = torch.zeros(3, 1, size_pop).to(device)
            o[2] = surfaces[aperture_index - 1].d
            o[0] = 1e-10
            d[0] = 1e-10
            d[2] = surfaces[aperture_index - 1].d - surfaces[aperture_index].d
            ray = Ray(o, d / d.square().sum(dim=0).sqrt(), wavelength)
            valid, ray_out = self._trace(ray, stop_ind=aperture_index, record=False)
            if not valid.any():
                raise Exception('ray trace is wrong!')
            o_out = ray_out.o
            d_out = ray_out.d
            enpp = (o_out[2] + (0 - o_out[0]) / d_out[0] * d_out[2]).squeeze(0)
            enpp[torch.isnan(enpp)] = 1  # TODO:这行代码很重要
            self.order1.enpp = enpp
            # 正向追迹得到孔阑半径
            first_x = ((self.basics.z_object - enpp) / self.basics.z_object) * (self.order1.enpd / 2)
            first_x[torch.isnan(first_x)] = 1
            first_x[first_x > self.basics.max_r] = self.basics.max_r
            o = torch.zeros(3, M, size_pop).to(device)
            d = torch.zeros(3, M, size_pop).to(device)

            o[0] = first_x.unsqueeze(0) * torch.linspace(0.7, 1, M).unsqueeze(1).to(device)
            d[0] = first_x.unsqueeze(0) * torch.linspace(0.7, 1, M).unsqueeze(1).to(device)
            d[2] = -self.basics.z_object
            ray = Ray(o, d / d.square().sum(dim=0).sqrt(), wavelength)
            valid, ray_out = self._trace(ray, stop_ind=aperture_index, record=False)
            item = ray_out.o[0].abs()
            item[~valid] = -1
            max_aper = item.max(dim=0).values
            if not valid.any():
                raise Exception('ray trace is wrong!')
            self.order1.valid_idx[(self.order1.effl <= 0) | (self.order1.enpp < self.basics.z_object) | (max_aper < 0) | (valid.sum(dim=0) < M//2)] = False
            return max_aper

    def calc_effl(self, wavelength=588.0, item=1e-4, approx=False):
        device = self.basics.device
        size_pop = self.basics.max_pop
        # 主光线与边缘光线在第一面的坐标
        o = torch.zeros(3, 1, size_pop).to(device)
        d = torch.zeros(3, 1, size_pop).to(device)
        o[0] = item
        o[2] = self.order1.surfaces[0].g(item, 0) - 0.1
        d[2] = 1
        # 生成相应光线
        ray = Ray(o, d, wavelength)
        time1 = time.time()
        valid, ray_out = self._trace(ray, record=False)
        # time2 = time.time()
        # print(time2-time1)
        if not valid.any():
            raise Exception('effl ray trace is wrong!')
        o_out = ray_out.o
        d_out = ray_out.d
        t = (0 - o_out[0]) / d_out[0]
        z_sensor = o_out[2] + t * d_out[2]
        t = (item - o_out[0]) / d_out[0]
        z_main = o_out[2] + t * d_out[2]
        effl = z_sensor - z_main
        return effl.squeeze(0)

    def calc_fov_samples(self, fov_max, pos=None):
        device = self.basics.device
        if pos is None:
            order = torch.linspace(0, 1, self.basics.fov_sample_num).to(device)
        else:
            order = torch.Tensor(pos).to(device)
        views = torch.arctan(order * math.tan(
            fov_max / 180 * math.pi)) / math.pi * 180
        return views, order

    def calc_enp_all_fov(self, wavelength=588.04):
        views = self.order1.fov_samples
        surfaces = self.order1.surfaces
        device = self.basics.device
        aperture_index = self.basics.aper_index

        self.order1.valid_idx_now = torch.ones(self.order1.valid_idx.sum()).to(device).bool()
        M = self.basics.enp_sample_num
        M2 = M ** 2
        valid_rate = 0.01

        vignetting = 1
        size_pop_now = self.order1.valid_idx.sum()
        ray_o = torch.zeros(3, M2 * self.basics.fov_sample_num, size_pop_now).to(device)
        ray_d = torch.zeros(3, M2 * self.basics.fov_sample_num, size_pop_now).to(device)
        for i in range(self.basics.fov_sample_num):
            fov = views[i]
            ray = self.sample_ray_common(wavelength, view=fov, M=M, mode='init')
            ray_o[:, i * M2:(i + 1) * M2] = ray.o
            ray_d[:, i * M2:(i + 1) * M2] = ray.d
        ray_all = Ray(ray_o, ray_d, wavelength)
        valid_all, ray_final_all, oss_all = self._trace(ray_all, record=True)
        for i in range(self.basics.fov_sample_num):
            valid = valid_all[i * M2:(i + 1) * M2]
            self.order1.valid_idx_now = self.order1.valid_idx_now & (valid.sum(dim=0) > (M2 * valid_rate))
        self.order1.valid_idx[self.order1.valid_idx.clone()] = self.order1.valid_idx_now.clone()
        valid_index = self.order1.valid_idx_now
        self.re_valid_func(valid_index)
        # resize
        item_oss = oss_all[..., self.order1.valid_idx_now]
        aper_p = item_oss[aperture_index + 1, 0]
        item_valid = (((aper_p - surfaces[aperture_index].r * vignetting) < 0) & (
                (aper_p + surfaces[aperture_index].r * vignetting) > 0)) & valid_all[..., valid_index]

        for i in range(len(surfaces)):
            if i != aperture_index:
                p1 = (item_oss[i + 1, 0] ** 2 + item_oss[i + 1, 1] ** 2) ** 0.5
                p1[~item_valid] = 0
                surfaces[i].r[surfaces[i].r > p1.max(dim=0).values] = p1.max(dim=0).values[
                    surfaces[i].r > p1.max(dim=0).values]

        enp_xyz = torch.zeros(5, views.size(0), valid_index.sum()).to(device)

        for i in range(self.basics.fov_sample_num):
            # valid = valid_all[i * (M * M):(i + 1) * (M * M), self.order1.valid_idx_now]
            valid = item_valid[i * M2:(i + 1) * M2]
            oss = oss_all[:, :, i * M2:(i + 1) * M2, self.order1.valid_idx_now]
            first_px = oss[0, 0]
            first_py = oss[0, 1]
            z = oss[0, 2, 0]

            first_px[~valid] = 10e6
            x1 = first_px.min(dim=0).values
            first_px[~valid] = -10e6
            x2 = first_px.max(dim=0).values
            first_py[~valid] = 10e6
            y1 = first_py.min(dim=0).values
            first_py[~valid] = -10e6
            y2 = first_py.max(dim=0).values

            enp_xyz[:, i] = torch.stack(
                ((x1 + x2) / 2, torch.zeros_like(x1), (x2 - x1) / 2 + 0.01, (y2 - y1) / 2 + 0.01, z), dim=0)

        return enp_xyz

    def sample_ray_common(self, wavelength, view=None, pos=None, M=15, mode='init'):
        M2 = M * M
        surface = self.order1.surfaces[0]
        device = self.basics.device
        angle = view / 180 * torch.pi  # 用弧度值代替度值数组
        size_pop_now = self.order1.valid_idx.sum()
        o = torch.zeros(3, M2, size_pop_now).to(device)
        origin = torch.zeros(3, M2, size_pop_now).to(device)
        origin[0] = (self.order1.enpp - self.basics.z_object) * torch.tan(angle)
        origin[2] = self.basics.z_object
        if mode == 'init':
            Mx = M2 // 2
            My = M2 - M2 // 2
            re_x = 0.2
            R = (self.order1.enpd / 2) * (1 + re_x)
            x_index = (torch.linspace(0, Mx - 1, Mx).to(device)).unsqueeze(1)
            y_index = (torch.linspace(0, My - 1, My).to(device)).unsqueeze(1)
            x = - R + 2 * R / (Mx - 1) * x_index
            y = - R + 2 * R / (My - 1) * y_index
            o[0, :Mx] = x
            o[1, Mx:] = y
            o[2] = self.order1.enpp
        if mode == 'rms':
            cen_x = pos[0]
            cen_y = pos[1]
            R_x = pos[2]
            R_y = pos[3]
            x_index = (torch.linspace(0, M - 1, M).to(device)).expand(M, M).T.flatten().unsqueeze(1)
            y_index = (torch.linspace(0, M - 1, M).to(device)).expand(M, M).flatten().unsqueeze(1)
            x = cen_x - R_x + 2 * R_x / (M - 1) * x_index
            y = cen_y - R_y + 2 * R_y / (M - 1) * y_index
            o[0] = x
            o[1] = y
            o[2] = pos[4]
        if mode == '2D':
            cen_x = pos[0]
            R = pos[2]
            x_index = (torch.linspace(0, M2 - 1, M2).to(device)).unsqueeze(1)
            x = cen_x - R + 2 * R / (M2 - 1) * x_index
            o[0] = x
            o[2] = pos[4]

        d = o - origin
        d[2][d[2] < 0] = 1
        d = d / d.square().sum(dim=0).sqrt()
        z_init = torch.clamp(surface.g(surface.r, 0) - 0.1, max=-1e-4)
        o = o + (z_init - o[2]) / d[2] * d

        return Ray(o, d, wavelength)  # 这里返回的是一个光线类

    def fit_constraint_func(self, x, bound, exponent=1):
        min_x = bound[0]
        max_x = bound[1]

        loss_con = torch.zeros_like(x)
        loss_con[x > max_x] = (x - max_x)[x > max_x] ** exponent
        loss_con[x < min_x] = (min_x - x)[x < min_x] ** exponent
        return loss_con * bound[2]

    def calc_fit_img(self, weight_relative=1, weight_ray=1, weight_color=0.25, weight_distort=1, flag='rms'):
        device = self.basics.device
        M = self.basics.enp_sample_num
        num_obj = len(self.basics.z_object_edof)
        views = self.order1.fov_samples
        if self.order1.opti_stage == 'single_wave':
            wave_all_data = [
                self.order1.wave_data["all"][(self.order1.now_gener + 1) % (len(self.order1.wave_data["all"]))]]
        else:
            wave_all_data = self.order1.wave_data["all"]
        size_pop_now = self.order1.valid_idx.sum()
        self.order1.valid_idx_now = torch.ones(self.order1.valid_idx.sum()).to(device).bool()
        M2 = M ** 2
        RMS_all = torch.zeros(num_obj, self.basics.fov_sample_num, len(wave_all_data), size_pop_now).to(device)
        vignetting_factors = torch.zeros_like(RMS_all)
        loss_ray_all = torch.zeros_like(RMS_all)
        spot_cen = torch.zeros_like(RMS_all)
        for num in range(num_obj):
            self.basics.z_object = self.basics.z_object_edof[num]

            for j, wave_len in enumerate(wave_all_data):
                ray_o = torch.zeros(3, M2 * self.basics.fov_sample_num, size_pop_now).to(device)
                ray_d = torch.zeros(3, M2 * self.basics.fov_sample_num, size_pop_now).to(device)
                for i in range(self.basics.fov_sample_num):
                    fov = views[i]
                    ray = self.sample_ray_common(wave_len, view=fov, pos=self.order1.enp_xyz[:, i], M=M, mode='rms')
                    ray_o[:, i * M2:(i + 1) * M2] = ray.o
                    ray_d[:, i * M2:(i + 1) * M2] = ray.d
                ray_all = Ray(ray_o, ray_d, wave_len)
                valid_all, ray_final_all, oss_all = self._trace(ray_all, record=True)
                for i in range(self.basics.fov_sample_num):
                    valid = valid_all[i * (M * M):(i + 1) * (M * M)]
                    ray_o = ray_final_all.o[:, i * (M * M):(i + 1) * (M * M)]
                    ray_d = ray_final_all.d[:, i * (M * M):(i + 1) * (M * M)]
                    oss = oss_all[:, :, i * (M * M):(i + 1) * (M * M)]

                    # 计算光线误差
                    img_gt = torch.zeros_like(oss[0])
                    img_gt[0] = -self.basics.img_r * self.order1.fov_pos[i]
                    img_gt[2] = self.order1.z_sensor

                    d_gt = img_gt - oss[0]
                    d_gt = d_gt / d_gt.square().sum(dim=0).sqrt()
                    d_real = oss.diff(dim=0)
                    d_real = (d_real.transpose(0, 1) / d_real.square().sum(dim=1).sqrt()).transpose(0, 1)
                    item_ray = torch.clamp(0.5 - (d_gt * d_real).sum(dim=1), min=0).sum(dim=0)
                    item_ray[~valid] = 0
                    loss_ray_all[num, i, j] = item_ray.sum(dim=0) / valid.sum(dim=0)
                    # 计算相对照度
                    ray_d_z = ray_d[2]
                    ray_d_xyz = (ray_d[0].square() + ray_d[1].square() + ray_d_z.square()).sqrt()
                    item_cos = ray_d_z / ray_d_xyz
                    item_cos[~valid] = 0
                    cos_img_angle = (item_cos ** 4).sum(dim=0) / valid.sum(dim=0)

                    vignetting_factors[num, i, j] = self.order1.enp_xyz[:, i][2] * self.order1.enp_xyz[:, i][
                        3] * valid.sum(dim=0) * cos_img_angle
                    t = (self.order1.z_sensor - ray_o[2]) / ray_d[2]
                    p_img = ray_o + t * ray_d
                    valid_now = torch.stack((valid, valid, valid), dim=0)
                    p_img[~valid_now] = 0
                    ps_xy = p_img[:2]
                    # 点列图
                    if flag == 'rms':
                        ps_xy[0][~valid] = 0
                        center_x = ps_xy[0].sum(dim=0)/valid.sum(dim=0)
                        ps_xy[0] = ps_xy[0] - center_x
                        spot_cen[num, i, j] = center_x
                    # 点对点
                    if flag == 'one':
                        ps_xy[0] = ps_xy[0] + self.basics.img_r * self.order1.fov_pos[i]
                    ps_xy[~valid_now[:2]] = 0  # we now use normalized ps
                    result = torch.sqrt(torch.sum(ps_xy ** 2, dim=1).sum(dim=0) / valid.sum(dim=0))
                    # 特定损失设置
                    RMS_all[num, i, j] = result
        # 计算畸变以及色差
        loss_color = torch.zeros(size_pop_now).to(device)
        loss_distort = torch.zeros(size_pop_now).to(device)
        if flag == 'rms':
            wave_num = spot_cen.size(2)
            if wave_num > 1:
                loss_color = (spot_cen[:, :, -1]-spot_cen[:, :, 0]).abs().mean(dim=0).mean(dim=0)
            for i in range(spot_cen.size(1)):
                img_weight = self.order1.fov_pos[i]
                if 0 < i < spot_cen.size(1) - 1:
                    distort = (spot_cen[:, i] + self.basics.img_r * img_weight) / (self.basics.img_r * img_weight)
                    loss_distort = loss_distort + self.fit_constraint_func(distort, self.basics.DISG, exponent=1).mean(dim=0).mean(dim=0)
                elif i == spot_cen.size(1) - 1:
                    distort = (spot_cen[:, i] + self.basics.img_r * img_weight) / (
                                self.basics.img_r * img_weight)
                    loss_distort = loss_distort + self.fit_constraint_func(distort, self.basics.RIMH, exponent=1).mean(dim=0).mean(dim=0)

        vignetting_factors_fin = vignetting_factors.mean(dim=2).mean(dim=0)
        self.order1.vignetting_factors = vignetting_factors_fin / vignetting_factors_fin.max(dim=0).values
        loss_relative = torch.clamp(0.4 - self.order1.vignetting_factors[-1], min=0)
        fit_all = RMS_all.mean(dim=0).mean(dim=0).mean(
            dim=0) + loss_relative * weight_relative + weight_ray * loss_ray_all.mean(dim=0).mean(dim=0).mean(dim=0) + \
                  weight_color * loss_color + weight_distort * loss_distort
        self.basics.z_object = self.basics.z_object_edof[num_obj // 2]
        return fit_all

    def calc_fit_constraint(self):

        surfaces = self.order1.surfaces
        device = self.basics.device
        loss_limits = []

        for i in range(len(surfaces) - 1):
            loss_border = torch.zeros_like(surfaces[i].r)
            now_material = self.basics.structure[i][2]
            # for j in range(6):
            weight = 1
            sag2 = surfaces[i + 1].surface(surfaces[i + 1].r * weight, 0.0)
            sag1 = surfaces[i].surface(surfaces[i].r * weight, 0.0)
            d_border = (surfaces[i + 1].d + sag2) - (surfaces[i].d + sag1)
            if now_material == 'G':
                loss_border += self.fit_constraint_func(d_border, self.basics.ETVAG, exponent=1)
            else:
                loss_border += self.fit_constraint_func(d_border, self.basics.ETVAA, exponent=1)

            loss_limits.append(loss_border)

        # 最后一面和像面距离，后截距
        sag1 = surfaces[-1].surface(surfaces[-1].r, 0.0)
        sag1[sag1 < 0] = 0
        d_border = self.order1.z_sensor - (surfaces[-1].d + sag1)
        loss_limits.append(self.fit_constraint_func(d_border, self.basics.BFFL, exponent=1))

        # 像面位置，也即是系统总长
        loss_limits.append(self.fit_constraint_func(self.order1.z_sensor, self.basics.TTVA, exponent=1))

        # 焦距物理约束
        loss_limits.append(self.fit_constraint_func(self.order1.effl, self.basics.EFFL, exponent=1))

        # 入瞳位置约束
        #
        # loss_enp = torch.zeros_like(enp_pos)
        # loss_enp[enp_pos < -surfaces[0].r * 0.01] = weight_enp
        # if self.basics.aper_index > 0:
        #     for i in range(self.basics.fov_sample_num - 1):
        #         item_enp = self.order1.enp_xyz[0, i + 1] - self.order1.enp_xyz[0, i]
        #         loss_enp[item_enp < surfaces[0].r * 0.01] = loss_enp[item_enp < surfaces[0].r * 0.01] + weight_enp
        # loss_limits.append(loss_enp)

        return sum(loss_limits)

    # ------------------------------------------------------------------------------------
    # 画图
    # ------------------------------------------------------------------------------------
    def draw_all(self, M=11, sort_index=None):
        num = self.order1.now_gener
        now_valid_index = torch.where(self.order1.valid_idx)[0]
        now_sort_index = torch.zeros_like(sort_index)
        for i in range(len(sort_index)):
            now_sort_index[i] = torch.nonzero(now_valid_index == sort_index[i])[0][0]
        self.re_valid_func(now_sort_index)
        self.order1.valid_idx[:] = False
        self.order1.valid_idx[sort_index] = True

        with torch.no_grad():
            # 光线追迹可视化
            for i in range(len(now_sort_index)):
                fit_item = self.order1.fit_all[sort_index[i]].item()
                ax, fig = self.plot_setup2D_with_trace_whole(self.order1.wave_data["main"], M=M, index=i)
                ax.axis('off')
                title_name = "ray_gen{num}_{count}_fit{fit}".format(num=num, count=self.basics.draw_num
                                                                    , fit=round(fit_item, 4))
                name = self.basics.demo_root + title_name + ".png"

                fig.savefig(name, bbox_inches='tight')
                plt.close()
                self.basics.draw_num = self.basics.draw_num + 1

    def spot_diagram(self, ps, show=True, x_lims=None, y_lims=None, color='b.', save_path=None):
        """
        画点列图。
        """
        units = 1
        spot_rms = float(self.rms(ps, units)[0])
        ps = ps.cpu().detach().numpy()[..., :2]
        ps_mean = np.mean(ps, axis=0)  # 质心
        ps = ps - ps_mean[None, ...]  # we now use normalized ps

        fig = plt.figure()
        ax = plt.axes()
        ax.plot(ps[..., 1], ps[..., 0], color)
        plt.gca().set_aspect('equal', adjustable='box')

        if x_lims is not None:
            plt.xlim(*x_lims)
        if y_lims is not None:
            plt.ylim(*y_lims)
        ax.set_aspect(1. / ax.get_data_ratio())
        units_str = '[mm]'
        plt.xlabel('x ' + units_str)
        plt.ylabel('y ' + units_str)
        plt.xticks(np.linspace(x_lims[0], x_lims[1], 11))
        plt.yticks(np.linspace(y_lims[0], y_lims[1], 11))
        # plt.grid(True)

        if save_path is not None:
            fig.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()

        return spot_rms

    def draw_points(self, ax, options, seq=range(3)):
        for surface in self.surfaces:
            points_world = self._generate_points(surface)
            ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)

    def generate_mesh(self, name=None, scale=1):
        points = []
        for surface in self.surfaces:
            p, b = self._generate_points(surface, with_boundary=True)
            del b
            points.append(scale * p.T)

        # TODO: 目前仅适用于两个曲面
        x = [points[i][:, 0] for i in range(2)]
        y = [points[i][:, 1] for i in range(2)]
        z = [points[i][:, 2] for i in range(2)]
        tris = [tri.Triangulation(x[i], y[i]) for i in range(2)]
        triangles = [tris[i].triangles for i in range(2)]

        X = np.hstack((x[0], x[1]))
        Y = np.hstack((y[0], y[1]))
        Z = np.hstack((z[0], z[1]))
        T = np.vstack((triangles[0], 1 + triangles[0].max() + triangles[1]))
        mesh = meshio.Mesh(np.stack((X, Y, Z), axis=-1), [("triangle", T)])
        if name is not None:
            mesh.write(name)
        return points

    def plot_setup2D(self, ax=None, fig=None, show=True, color='k', with_sensor=True, index=None):
        surfaces = self.order1.surfaces
        device = self.basics.device
        materials = self.order1.materials
        aperture_distance = surfaces[self.basics.aper_index].d
        aperture_radius = surfaces[self.basics.aper_index].r
        if ax is None and fig is None:
            fig, ax = plt.subplots(figsize=(20, 15))
        else:
            show = False

        # to world coordinate

        def plot(axx, zz, xx, colors, line_width=5.0):
            p = torch.stack((xx, torch.zeros_like(xx, device=device), zz), dim=1).cpu().detach().numpy()
            axx.plot(p[..., 2], p[..., 0], colors, linewidth=line_width)

        def draw_aperture(axx, d, R, colors):
            N = 3
            APERTURE_WEDGE_LENGTH = 0.05 * R  # [mm]
            APERTURE_WEDGE_HEIGHT = 0.15 * R  # [mm]

            # wedge length
            z_aperture = torch.linspace(d - APERTURE_WEDGE_LENGTH, d + APERTURE_WEDGE_LENGTH, N, device=device)
            x_aperture = -R * torch.ones(N, device=device)
            plot(axx, z_aperture, x_aperture, colors)
            x_aperture = R * torch.ones(N, device=device)
            plot(axx, z_aperture, x_aperture, colors)

            # wedge height
            z_aperture = d * torch.ones(N, device=device)
            x_aperture = torch.linspace(R, R + APERTURE_WEDGE_HEIGHT, N, device=device)
            plot(axx, z_aperture, x_aperture, colors)
            x_aperture = torch.linspace(-R - APERTURE_WEDGE_HEIGHT, -R, N, device=device)
            plot(axx, z_aperture, x_aperture, colors)

        if len(surfaces) == 1:  # if there is only one surface, then it has to be the aperture
            draw_aperture(ax, aperture_distance[index], aperture_radius[index], color)
        else:
            if with_sensor:
                surfaces.append(Aspheric(torch.zeros_like(self.order1.z_sensor), self.order1.z_sensor,
                                         torch.ones_like(self.order1.z_sensor) * self.basics.img_r))

            sag = surfaces[self.basics.aper_index].g(surfaces[self.basics.aper_index].r, 0)
            if surfaces[self.basics.aper_index].c[index].abs()<1e-6:
                draw_aperture(ax, (aperture_distance + sag)[index].item(), aperture_radius[index], color)
            # 绘制透镜表面
            for i, s in enumerate(surfaces):
                if (i + 1) < len(surfaces) and materials[i].n[index] < 1.0003 and materials[i + 1].n[
                    index] < 1.0003 and i == self.basics.aper_index:
                    continue
                # if (i + 1) < len(surfaces) and materials[i].n[index] > 1.0003 and materials[i + 1].n[index] > 1.0003:
                #     r1 = s.r[index]
                #     r2 = surfaces[i + 1].r[index]
                #     item_r = max(r1, r2)
                #     s.r[index] = item_r
                #     surfaces[i + 1].r[index] = item_r
                r = torch.linspace(-s.r[index], s.r[index], s.APERTURE_SAMPLING, device=device)  # aperture sampling
                z = torch.zeros_like(r)
                for j in range(len(z)):
                    z[j] = s.surface_with_offset(r[j], 0)[index]
                plot(ax, z, r, color)

            # 绘制边界
            s_prev = []
            i = 0
            while i < len(surfaces):
                if i < len(surfaces) - 1 and materials[i].n[index] < 1.0003 and materials[i + 1].n[index] > 1.0003:
                    s = surfaces[i]
                    r_all = [s.r[index]]
                    sag_all = [s.surface_with_offset(s.r[index], 0.0)[index].squeeze(0)]
                    i += 1
                    while materials[i].n[index] > 1.0003:
                        s = surfaces[i]
                        r_all.append(s.r[index])
                        sag_all.append(s.surface_with_offset(s.r[index], 0.0)[index].squeeze(0))
                        i += 1
                    r_max = torch.Tensor(r_all).max()
                    for j in range(len(sag_all)):
                        if r_all[j] < r_max:
                            z = torch.stack((sag_all[j], sag_all[j]))
                            x = torch.Tensor([r_all[j], r_max]).to(device)
                            plot(ax, z, x, color)
                            plot(ax, z, -x, color)
                    z = torch.stack((min(sag_all), max(sag_all)))
                    x = torch.Tensor([r_max, r_max]).to(device)
                    plot(ax, z, x, color)
                    plot(ax, z, -x, color)
                else:
                    i += 1

            # 移除传感器平面
            if with_sensor:
                surfaces.pop()

        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlabel('z [mm]')
        # plt.ylabel('r [mm]')
        # show = True
        # if show:
        #     plt.show()
        return ax, fig

    def plot_setup2D_with_trace_single(self, views, wavelength, M=11):
        colors_list = 'bgrymckbgrymckbgrymck'
        ax, fig = self.plot_setup2D(show=False)
        for i, view in enumerate(views):
            ray = self.sample_ray_2D_common(wavelength, view=view, M=M, R=self.entrance_pos[i][2],
                                            cen=self.entrance_pos[i][0:2])
            oss = self.trace_to_sensor(ray)
            ax, fig = self.plot_ray_traces(oss, ax=ax, fig=fig, color=colors_list[i])
        return ax, fig

    def plot_setup2D_with_trace_whole(self, wavelength, M=11, index=None, keep_invalid=False):
        surfaces = self.order1.surfaces
        colors_list = ['m', 'b', 'c', 'g', 'y', 'r', 'k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        views = self.order1.fov_samples
        ax, fig = self.plot_setup2D(show=False, index=index)

        for i, view in enumerate(views):
            item = self.order1.enp_xyz.clone()

            ray = self.sample_ray_common(wavelength, view=view, M=M, pos=item[:, i], mode='2D')
            valid, ray_out, oss = self._trace(ray, record=True)
            # 与传感器平面相交
            t = (self.order1.z_sensor - ray_out.o[2]) / ray_out.d[2]
            p = ray_out(t)
            oss = torch.cat((oss, p.unsqueeze(0)), dim=0)[..., index]

            if surfaces[0].surface(surfaces[0].r, 0.0)[index] < 0:
                oss = oss[1:]
            if not keep_invalid:
                oss = oss[..., valid[..., index]]
            oss_now = oss[..., [0, oss.size(2)//2, -1]]
            ax, fig = self.plot_ray_traces(oss_now, ax=ax, fig=fig, color=colors_list[i % len(colors_list)])
        return ax, fig

    # TODO: modify the tracing part to include oss
    def plot_ray_traces(self, oss, ax=None, fig=None, color='b', show=True, p=None, valid_p=None, line_width=4.0):
        """
        Plot all ray traces (oss).
        """
        device = self.basics.device
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D(show=False)
        else:
            show = False
        for i in range(oss.size(-1)):
            os = oss[..., i].T
            x = os[0]
            z = os[2]

            # to world coordinate
            o = torch.stack((x, torch.zeros_like(x, device=device), z), dim=0).cpu().detach().numpy()
            z = o[2].flatten()
            x = o[0].flatten()

            if p is not None and valid_p is not None:
                if valid_p[i]:
                    x = np.append(x, p[i, 0])
                    z = np.append(z, p[i, 2])

            ax.plot(z, x, color, linewidth=line_width)

        if show:
            plt.show()
        else:
            plt.close()
        return ax, fig

    def calc_RMS(self, ray=None, calc_cen=False):
        valid, ray_out = self._trace(ray)
        # 与传感器平面相交
        t = (self.order1.z_sensor - ray_out.o[2]) / ray_out.d[2]
        p_img = ray_out(t)
        # 通过RMS直径的1.5倍(经验值)作为psf范围
        RMS, p_cen = self.rms(p_img, valid=valid)
        if calc_cen:
            return RMS, p_cen
        else:
            return RMS

    @staticmethod
    def rms(ps, valid=None, ceo=False, mode='3D'):
        if mode == '2D':
            valid_x = valid.clone()
            ps_x = ps[0]
            valid_x[ps_x == 0] = False
            ps_x[~valid_x] = 0
            ps_mean = ps_x.sum(dim=0) / valid_x.sum(dim=0)
            ps_x = ps_x - ps_mean
            ps_x[~valid_x] = 0
            result_x = torch.sqrt(torch.sum(ps_x ** 2, dim=0) / valid_x.sum(dim=0))
            valid_y = valid.clone()
            ps_y = ps[1]
            valid_y[ps_y == 0] = False
            ps_y[~valid_y] = 0
            result_y = torch.sqrt(torch.sum(ps_y ** 2, dim=0) / valid_y.sum(dim=0))
            result = result_y + result_x
        else:
            valid_all = torch.stack((valid, valid, valid), dim=0)
            ps[~valid_all] = 0
            ps_xy = ps[:2]
            ps_mean = (ps_xy.sum(dim=1) / valid.sum(dim=0))
            ps_cen = ps_xy - ps_mean.unsqueeze(1)
            ps_cen[~valid_all[:2]] = 0  # we now use normalized ps
            result = torch.sqrt(torch.sum(ps_cen ** 2, dim=1).sum(dim=0) / valid.sum(dim=0))
        result[torch.isnan(result)] = 1e02
        return result, ps_mean[0]

    def calc_max_fov(self, wavelength=588.0, M=1001, resume=False, r_ex=0.2, min_fov=5):
        # 运行时间计算
        surfaces = self.order1.surfaces
        device = self.basics.device
        aperture_index = self.basics.aper_index
        img_r = self.basics.img_r
        size_pop = self.simulation.b_size
        if resume:
            for i in range(len(surfaces) - 1):
                if i != aperture_index:
                    item = (1 + self.order1.surfaces[i].k) * (self.order1.surfaces[i].c ** 2)
                    item[item <= 0] = 1e-6
                    surfaces[i].r = (1 / item) ** 0.5
                    surfaces[i].r[surfaces[i].r > self.basics.max_r] = self.basics.max_r
        sag = surfaces[-1].surface(surfaces[-1].r, 0.0)
        r = surfaces[-1].r
        d = surfaces[-1].d
        d_target = d + sag
        R1 = -(r + (r - img_r) * sag / (self.order1.z_sensor - d_target))
        R1[R1 > -r * (1 - r_ex)] = (-r * (1 - r_ex))[R1 > -r * (1 - r_ex)]
        R1[R1 < -r * (1 + r_ex)] = (-r * (1 + r_ex))[R1 < -r * (1 + r_ex)]
        R2 = r + (r + img_r) * sag / (self.order1.z_sensor - d_target)
        R2[R2 > r * (1 - r_ex)] = (r * (1 - r_ex))[R2 > r * (1 - r_ex)]
        R2[R2 < r * (1 + r_ex)] = (r * (1 + r_ex))[R2 < r * (1 + r_ex)]
        o = torch.zeros(3, M, size_pop).to(device)
        origin = torch.zeros(3, M, size_pop).to(device)
        origin[0] = -img_r
        origin[2] = self.order1.z_sensor
        o[2] = surfaces[-1].d

        o[0] = R1.unsqueeze(0) + (R2 - R1).unsqueeze(0) / (M - 1) * torch.linspace(0, M - 1, M).to(device).unsqueeze(1)
        d = o - origin
        ray = Ray(o, d / d.square().sum(dim=0).sqrt(), wavelength)
        valid, ray_final = self._trace(ray, record=False)
        if not valid.any():
            raise Exception('ray trace is wrong!')
        t_img = (self.basics.z_object - ray_final.o[2]) / ray_final.d[2]
        p_img = ray_final(t_img)
        p_img[0][~valid] = 0
        # 严格一点，将valid少一点的和视场角太小的也置为false
        item_fov = (torch.arctan(
            p_img[0].sum(dim=0) / valid.sum(dim=0) / (self.order1.enpp - self.basics.z_object))) / torch.pi * 180
        self.order1.valid_idx[(item_fov < min_fov) & (valid.sum(dim=0) < (M * 0.05))] = False
        fov_max = item_fov[self.order1.valid_idx]
        # 开始re_valid的地方
        self.re_valid_func(self.order1.valid_idx)

        return fov_max.abs()

    def re_valid_func(self, valid_index):

        surfaces = self.order1.surfaces
        materials = self.order1.materials
        for i in range(len(surfaces)):
            surface = surfaces[i]
            surface.c = surface.c[valid_index]
            surface.d = surface.d[valid_index]
            surface.r = surface.r[valid_index]
            surface.k = surface.k[valid_index]
            if surface.ai is not None:
                surface.ai = surface.ai[:, valid_index]
        for i in range(len(materials)):
            mater = materials[i]
            mater.n = mater.n[valid_index]
            mater.V = mater.V[valid_index]
            if mater.A is not None:
                mater.A = mater.A[valid_index]
            if mater.B is not None:
                mater.B = mater.B[valid_index]
            if mater.n_real is not None:
                mater.n_real = mater.n_real[valid_index]
            if mater.name_idx is not None:
                mater.name_idx = mater.name_idx[valid_index]
        self.order1.enpp = self.order1.enpp[valid_index]
        self.order1.enpd = self.order1.enpd[valid_index]
        self.order1.effl = self.order1.effl[valid_index]
        self.order1.z_sensor = self.order1.z_sensor[valid_index]
        if self.order1.enp_xyz is not None:
            self.order1.enp_xyz = self.order1.enp_xyz[..., valid_index]
        if self.order1.vignetting_factors is not None:
            self.order1.vignetting_factors = self.order1.vignetting_factors[..., valid_index]
        if self.order1.aperture_max is not None:
            self.order1.aperture_max = self.order1.aperture_max[..., valid_index]
        if self.order1.aper_flag is not None:
            self.order1.aper_flag = self.order1.aper_flag[..., valid_index]

    def re_init(self):
        if self.order1.effl is not None:
            self.order1.effl = None
        if self.order1.enpp is not None:
            self.order1.enpp = None
        if self.order1.enp_xyz is not None:
            self.order1.enp_xyz = None
        if self.order1.vignetting_factors is not None:
            self.order1.vignetting_factors = None
        if self.order1.fov_samples is not None:
            self.order1.fov_samples = None
        if self.basics.distort_all is not None:
            self.basics.distort_all = []

    def resize_self(self, wavelength=588.0, view=None, M=1001, vignetting=1, re_valid=True, r_ex=0.3, valid_rate=0.05):
        size_pop = self.order1.valid_idx.sum()
        radians = view / 180 * torch.pi
        surfaces = self.order1.surfaces
        device = self.basics.device
        aperture_index = self.basics.aper_index
        self.order1.valid_idx_now = torch.ones(self.order1.valid_idx.sum()).to(device).bool()
        R = surfaces[0].r  # [mm]
        sag = (torch.tan(radians) * surfaces[0].surface(surfaces[0].r, 0.0)).abs()
        sag[sag > R * r_ex] = (R * r_ex)[sag > R * r_ex]
        R1 = -R - sag
        R2 = R + sag
        o = torch.zeros(3, M, size_pop).to(device)
        origin = torch.zeros(3, M, size_pop).to(device)
        o[0] = R1.unsqueeze(0) + (R2 - R1).unsqueeze(0) / (M - 1) * torch.linspace(0, M - 1, M).to(device).unsqueeze(1)
        origin[0] = (self.order1.enpp - self.basics.z_object) * torch.tan(radians)
        origin[2] = self.basics.z_object
        d = o - origin
        ray = Ray(o, d / d.square().sum(dim=0).sqrt(), wavelength)
        valid, _, oss = self._trace(ray, record=True)
        if not valid.any():
            raise Exception('ray trace is wrong!')
        self.order1.valid_idx_now = self.order1.valid_idx_now & (valid.sum(dim=0) > (M * valid_rate))
        self.order1.valid_idx[self.order1.valid_idx.clone()] = self.order1.valid_idx_now.clone()
        valid_index = self.order1.valid_idx_now
        if re_valid:
            oss = oss[..., valid_index]
            self.re_valid_func(valid_index)

        aper_p = oss[aperture_index + 1, 0]
        item_valid = (((aper_p - surfaces[aperture_index].r * vignetting) < 0) & (
                (aper_p + surfaces[aperture_index].r * vignetting) > 0)) & valid[..., valid_index]
        for i in range(len(surfaces)):
            if i != aperture_index:
                p1 = oss[i + 1, 0].abs()
                p1[~item_valid] = 0
                surfaces[i].r = p1.max(dim=0).values + 0.1

    # 在透镜前，计算得到近轴放大率，注意使用64位
    def calc_room_rate(self, wavelength=588.0, view=0.0):
        surfaces = self.order1.surfaces
        device = self.basics.device
        angle = np.radians(view)
        if angle == 0:
            angle += 1e-8
        # 主光线与边缘光线在第一面的坐标
        first_x = torch.Tensor([1e-10, 1e-10]).to(device).double()
        ones = torch.ones_like(first_x)
        zeros = torch.zeros_like(first_x)
        # o为在孔阑面的空间坐标
        o = torch.stack((first_x, zeros, zeros))
        # origin为在物面的坐标
        obj_x = torch.Tensor([0, 1e-10]).to(device).double()
        origin = torch.stack([obj_x, zeros, self.basics.z_object * ones])
        d = o - origin
        d = torch.div(d, d.norm(2, 0))
        d = d.T
        o = o.T
        # 生成相应光线
        ray = Ray(o, d, wavelength)
        # 输出光线，第一根为主光线，第二根为边缘光线
        _, ray_out = self._trace(ray, stop_ind=None, record=False)
        # 近轴放大率计算
        # 通过主光线求得近轴像位置
        o_out = ray_out.o[0]
        d_out = ray_out.d[0]
        t = (0 - o_out[0]) / d_out[0]
        d_sensor = o_out[2] + t * d_out[2]
        if self.order1.z_sensor is None:
            self.order1.z_sensor = d_sensor  # 更正位置
        # 通过近轴光线求得近轴像高度
        o_out = ray_out.o[1]
        d_out = ray_out.d[1]
        t = (d_sensor - o_out[2]) / d_out[2]
        img_r = o_out[0] + t * d_out[0]
        zoom_rate_pari = img_r / 1e-10
        return zoom_rate_pari.float()

    def calc_exit_pupil(self):
        # 孔阑中心反向追迹，得到入瞳位置
        surfaces = self.order1.surfaces
        device = self.basics.device
        aperture_index = self.basics.aper_index
        wavelength = self.order1.wave_data["main"]
        if aperture_index >= len(surfaces) - 1:
            expp = surfaces[aperture_index].d
        else:
            o = torch.Tensor([1e-10, 0, surfaces[aperture_index + 1].d]).unsqueeze(0).to(device).double()
            d = torch.Tensor([1e-10, 0, surfaces[aperture_index + 1].d - surfaces[aperture_index].d]).unsqueeze(0).to(
                device).double()
            ray = Ray(o, d / d.square().sum().sqrt(), wavelength)
            valid, ray_out = self._trace(ray, record=False)
            if not valid[0]:
                raise Exception('ray trace is wrong!')
            o_out = ray_out.o[0]
            d_out = ray_out.d[0]
            expp = (o_out[2] + (0 - o_out[0]) / d_out[0] * d_out[2]).item()
        self.order1.expp = expp

    def sample_ray_2D_common(self, wavelength, view=None, M=15, pos=None, mode='x'):
        if pos.dim() < 2:
            pos = pos.unsqueeze(1)
        device = self.basics.device
        angle = view / 180 * torch.pi  # 用弧度值代替度值数组
        cen_x = pos[0].unsqueeze(0)
        R_x = pos[2].unsqueeze(0)
        x = cen_x - R_x + 2 * R_x / (M - 1) * torch.linspace(0, M - 1, M).to(device).unsqueeze(1)
        size_pop_now = x.size(1)
        # 主光线与边缘光线在第一面的坐标
        o = torch.zeros(3, M, size_pop_now).to(device)
        origin = torch.zeros(3, M, size_pop_now).to(device)
        if mode == 'x':
            o[0] = x
        else:
            o[1] = x
        enpp_item = self.order1.enpp
        origin[0] = (enpp_item - self.basics.z_object) * torch.tan(angle)
        origin[2] = self.basics.z_object
        # 得到光线方向矢量d，2D方向改为3D方向，在x方向采样得到psf
        d = o - origin
        return Ray(o, d / d.square().sum(dim=0).sqrt(), wavelength)

    def write_optics_data(self, acc_idx, save_path='optics_data.xlsx', round_num=4):
        surfaces = self.order1.surfaces
        materials = self.order1.materials

        data = pd.read_excel(save_path).values
        i = 0
        order = -1
        while i < len(data):
            if isinstance(data[i, 0], str) and data[i, 0][:3] == 'num':
                order = int(data[i, 0][4:])
            i += 1
        order += 1

        for count in range(len(acc_idx)):
            data_now = np.zeros((len(surfaces) + 5, 20)).astype(object)

            data_now[1, 0] = 'num_' + str(order)
            data_now[1, 1] = 'fov_' + str(self.basics.fov)
            data_now[1, 2] = 'fnum_' + str(self.basics.fnum)
            data_now[1, 3] = 'aper_' + str(self.basics.aper_index)
            data_now[1, 4] = 'wave_' + str(self.order1.wave_data['all'])
            data_now[2] = np.array(['#', 'Type', 'Comment', 'Radius', 'Thickness', 'Material', 'Coating',
                                    'Semi-Diameter', 'Chip Zone', 'Mech Semi-Dia', 'Conic', 'TCE', 'Par 1',
                                    'Par 2', 'Par 3', 'Par 4', 'Par 5', 'Par 6', 'Par 7', 'Par 8'])
            data_now[3, 0] = 0
            data_now[3, 1] = 'STANDARD'
            data_now[-1, 0] = len(surfaces) + 1
            data_now[-1, 1] = 'STANDARD'
            data_now[3, 4] = abs(self.basics.z_object)
            now_idx = self.order1.valid_idx[:acc_idx[count]].sum()
            fit_now = self.order1.fit_all[self.order1.valid_idx][now_idx].item()
            data_now[1, 5] = 'fit_' + str(fit_now)

            for i in range(len(surfaces)):
                data_now[i + 4, 0] = i + 1
                now_structure = self.order1.now_structures[i]
                data_now[i + 4, 7] = surfaces[i].r[now_idx].item()
                if now_structure[2] == 'G':
                    if self.order1.use_real_glass:
                        # name = materials[i + 1].MATERIAL_TABLE.name[materials[i + 1].name_idx[now_idx]]
                        # data_now[i + 4, 5] = name
                        data_now[i + 4, 5] = str(materials[i + 1].n[now_idx].item()) + ';' + str(
                            materials[i + 1].V[now_idx].item()) + ';' + str(0)
                    else:
                        data_now[i + 4, 5] = str(materials[i + 1].n[now_idx].item()) + ';' + str(
                            materials[i + 1].V[now_idx].item()) + ';' + str(0)

                if i == len(surfaces) - 1:
                    data_now[i + 4, 4] = self.order1.z_sensor[now_idx].item() - surfaces[i].d[now_idx].item()
                else:
                    data_now[i + 4, 4] = surfaces[i + 1].d[now_idx].item() - surfaces[i].d[now_idx].item()
                if now_structure[0] == 'S':
                    c_item = surfaces[i].c[now_idx].item()
                    if c_item == 0:
                        c_item += 1e-10
                    data_now[i + 4, 3] = 1 / c_item
                    item = int(now_structure[1])
                    data_now[i + 4, 1] = 'STANDARD'
                    if item >= 1:
                        data_now[i + 4, 10] = surfaces[i].k[now_idx].item()
                    if item >= 2:
                        data_now[i + 4, 1] = 'EVENASPH'
                        for j in range(item - 1):
                            data_now[i + 4, 13 + j] = surfaces[i].ai[j, now_idx].item()
            if len(data) == 0:
                data = data_now.copy()
            else:
                if data.shape[1] < data_now.shape[1]:
                    data = np.concatenate((data, np.zeros((data.shape[0], data_now.shape[1] - data.shape[1]))), axis=1)
                data = np.concatenate((data, data_now))
            order += 1
        df = pd.DataFrame(data)
        df.to_excel(save_path, index=False)

    @staticmethod
    def read_data(read_path, device, iteration=8):
        surfaces = []
        materials = [Material('air')]
        with open(read_path, 'r', encoding='utf-8') as f:  # 使用with open()新建对象f
            contents = f.readlines()
            for i in range(len(contents)):
                if contents[i] == 'iteration:{num}\n'.format(num=iteration):
                    while contents[i + 2] != 'end\n':
                        content = str.split(contents[i + 2])
                        if len(content) <= 7:
                            surfaces.append(
                                Aspheric(float(content[4]), float(content[1]), c=float(content[2]), k=float(content[3]),
                                         device=device))
                        else:
                            ai = []
                            for ii in range(7, len(content)):
                                ai.append(torch.Tensor([float(content[ii])])[0].to(device))
                            surfaces.append(
                                Aspheric(float(content[4]), float(content[1]), c=float(content[2]), k=float(content[3]),
                                         ai=ai, device=device))
                        materials.append(Material(data=[float(content[5]), float(content[6])]))
                        i += 1
                    break
        f.close()
        return surfaces, materials

    def init_sel(self, div_rate=0.1, draw=True, max_num=200):
        device = self.basics.device
        contents = []
        aper = []
        data = []
        fit = []
        idx = []
        for item in range(len(self.path.save_root) - 1):
            read_path = self.path.save_root[item]
            with open(read_path, 'r', encoding='utf-8') as f:  # 使用with open()新建对象f
                contents = contents + f.readlines()
            f.close()

        i = 0
        while i < len(contents):
            if contents[i][:3] == 'num':
                idx.append(i)
                aper_now = int(str.split(contents[i + 1])[-1][9:])
                aper.append(aper_now)
                fit.append(float(str.split(contents[i + 1])[-2][4:]))
                i += 2
                pre_data = []
                while contents[i] != 'end\n':
                    content = np.array(str.split(contents[i]))
                    content[content == 'inf'] = math.inf
                    pre_data.append(content)
                    i += 1
                pre_data = np.array(pre_data).astype(float)
                surface_num = pre_data.shape[0]
                c = pre_data[:, 1]
                d = pre_data[:, 2]
                n = pre_data[:, 3]
                V = pre_data[:, 4]
                mater_idx = n[:surface_num - 1] > 1.003
                c2 = ((1 / c[c < math.inf]) - self.constraint.c[0]) / (self.constraint.c[1] - self.constraint.c[0])
                d2 = np.zeros(surface_num - 1)
                d2[mater_idx] = (d[:surface_num - 1][mater_idx] - self.constraint.thi_glass_cen[0]) / (
                        self.constraint.thi_glass_cen[1] - self.constraint.thi_glass_cen[0])
                d2[~mater_idx] = (d[:surface_num - 1][~mater_idx] - self.constraint.thi_air_cen[0]) / (
                        self.constraint.thi_air_cen[1] - self.constraint.thi_air_cen[0])
                n2 = (n[:surface_num - 1][mater_idx] - self.constraint.material_n[0]) / (
                        self.constraint.material_n[1] - self.constraint.material_n[0])
                V2 = (V[:surface_num - 1][mater_idx] - self.constraint.material_V[0]) / (
                        self.constraint.material_V[1] - self.constraint.material_V[0])
                data.append(np.array(c2.tolist() + d2.tolist() + n2.tolist() + V2.tolist()))
            i += 1
        row_per = round(len(contents) / len(data))
        fit_sorted_idx = (torch.Tensor(fit).to(device).sort()[1]).detach().cpu().numpy()
        data_pre = np.array(data)
        idx_final = []
        fit_final = []
        data_final = []
        aper_final = []
        for i in range(len(fit_sorted_idx)):
            idx_now = fit_sorted_idx[i]
            data_now = data_pre[idx_now]
            fit_now = fit[idx_now]
            aper_now = aper[idx_now]
            acc_flag = False
            if i == 0:
                acc_flag = True
            else:
                aper_idx = (np.array(aper_final) - aper_now == 0)
                if sum(aper_idx) == 0:
                    acc_flag = True
                elif (np.sqrt(np.sum(np.square(np.array(data_final) - data_now), axis=1)))[aper_idx].min() / (
                        len(data_now) ** 0.5) > div_rate:
                    acc_flag = True
            if acc_flag:
                idx_final.append(idx[idx_now])
                data_final.append(data_now)
                aper_final.append(aper_now)
                fit_final.append(fit_now)
        contents_final = []
        if len(idx_final) > max_num:
            idx_final = idx_final[:max_num]
        for i in range(len(idx_final)):
            idx = idx_final[i]
            item = contents[idx: idx + row_per]
            item[0] = 'num:' + str(i) + '\n'
            for j in range(len(item)):
                contents_final.append(item[j])
        with open(self.path.save_root[-1], 'a', encoding='utf-8') as f:
            for i in range(len(contents_final)):
                f.write(contents_final[i])
        f.close()

    def trace_valid(self, ray):
        """
        追迹光线以查看它们是否与传感器平面相交。
        """
        valid = self._trace(ray)[1]
        return valid

    @staticmethod
    def _refract(wi, n, eta, approx=False):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://physics.stackexchange.com/a/436252/104805
        """
        n = n
        wi = wi
        eta_ = eta
        cosi = torch.sum(wi * n, dim=0)

        if approx:
            tmp = 1. - eta ** 2 * (1. - cosi)
            valid = tmp > 0.
            wt = tmp * n + eta_ * (wi - cosi * n)
        else:
            cost2 = 1. - (1. - cosi ** 2) * eta ** 2

            # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid NaN grad at cost2==0.
            valid = (cost2 > 0.0001) & (cosi > 0.0001)
            cost2 = torch.clamp(cost2, min=1e-8)
            tmp = torch.sqrt(cost2)

            # here we do not have to do normalization because if both wi and n are normalized,
            # then output is also normalized.
            wt = tmp * n + eta_ * (wi - cosi * n)
        return valid, wt

    def trace_to_sensor(self, ray, record=False):
        """
        光线追迹，使其与传感器平面相交。
        """
        oss = None
        if record:
            valid, ray_out, oss = self._trace(ray, record=record)
        else:
            valid, ray_out = self._trace(ray, record=record)
        # 与传感器平面相交
        t = (self.order1.z_sensor - ray_out.o[2]) / ray_out.d[2]
        p = ray_out(t)
        if record:
            return oss, p
        else:
            return p

    def _trace(self, ray, start_ind=None, stop_ind=None, record=False):
        if stop_ind is None:
            stop_ind = len(self.order1.surfaces) - 1  # 如果没有设置是否在哪停下，最后一面作为stop
        if start_ind is None:
            start_ind = 0
        is_forward = (ray.d[2] > 0).all()  # 确认光线是否正向传播
        if is_forward:
            return self._forward_tracing(ray, start_ind, stop_ind, record)
        else:
            return self._backward_tracing(ray, start_ind, stop_ind, record)

    def _forward_tracing(self, ray, start_ind, stop_ind, record):
        wavelength = ray.wavelength
        surfaces = self.order1.surfaces
        materials = self.order1.materials
        device = self.basics.device
        oss = torch.ones(stop_ind + 2, ray.o.shape[0], ray.o.shape[1], ray.o.shape[2]).to(device)
        if record:
            oss[0] = ray.o
        valid = torch.ones(ray.o.shape[1], ray.o.shape[2], device=device).bool()
        for i in range(start_ind, stop_ind + 1):
            # 两个表面折射率之比
            eta = materials[i].ior(wavelength) / materials[i + 1].ior(wavelength)
            # 光线与透镜表面相交，p为与透镜表面交点，
            valid_o, p = surfaces[i].ray_surface_intersection(ray, valid)
            valid = valid & valid_o
            if not valid.any():
                break
            # item2 = p.transpose(0, 2)[valid.transpose(0, 1)]
            # # 异常值范围限制，防止计算报错
            # p2 = torch.stack((torch.clamp(p[0], min=item2[..., 0].min().item(), max=item2[..., 0].max().item()),
            #                   torch.clamp(p[1], min=item2[..., 1].min().item(), max=item2[..., 1].max().item()),
            #                   torch.clamp(p[2], min=item2[..., 2].min().item(), max=item2[..., 2].max().item())),
            #                  dim=0)
            p2 = p
            # 得到透镜表面法线
            n = surfaces[i].normal(p2[0], p2[1])
            valid_d, d = self._refract(ray.d, -n, eta)
            # 检验有效性
            valid_d[d[2] <= 0] = False
            valid = valid & valid_d
            if not valid.any():
                break
            ray.o = p2
            ray.d = d
            # 更新光线 {o,d}
            if record:
                oss[i + 1] = ray.o

        if record:
            valid[(oss.diff(dim=0)[:, 2] < 0).sum(dim=0) > 0] = False
            return valid, ray, oss
        else:
            return valid, ray

    def _backward_tracing(self, ray, start_ind, stop_ind, record):
        wavelength = ray.wavelength
        surfaces = self.order1.surfaces
        materials = self.order1.materials
        device = self.basics.device
        oss = torch.ones(stop_ind + 2, ray.o.shape[0], ray.o.shape[1], ray.o.shape[2]).to(device)
        if record:
            oss[-1] = ray.o
        valid = torch.ones(ray.o.shape[1], ray.o.shape[2], device=device).bool()
        for i in np.flip(range(start_ind, stop_ind + 1)):
            surface = surfaces[i]
            eta = materials[i + 1].ior(wavelength) / materials[i].ior(wavelength)
            # ray intersecting surface
            valid_o, p = surface.ray_surface_intersection(ray, valid)

            valid = valid & valid_o
            if not valid.any():
                break

            # 异常值范围限制，防止计算报错
            item2 = p.transpose(0, 2)[valid.transpose(0, 1)]
            # 异常值范围限制，防止计算报错
            p2 = torch.stack((torch.clamp(p[0], min=item2[..., 0].min().item(), max=item2[..., 0].max().item()),
                              torch.clamp(p[1], min=item2[..., 1].min().item(), max=item2[..., 1].max().item()),
                              torch.clamp(p[2], min=item2[..., 2].min().item(), max=item2[..., 2].max().item())),
                             dim=0)
            n = surfaces[i].normal(p2[0], p2[1])
            valid_d, d = self._refract(ray.d, n, eta)  # backward: no need to revert the normal
            # check validity
            valid = valid & valid_d
            if not valid.any():
                break
            # update ray {o,d}
            ray.o = p2
            ray.d = d
            if record:
                oss[i] = ray.o
        if record:
            return valid, ray, oss
        else:
            return valid, ray
