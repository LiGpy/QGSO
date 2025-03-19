import os
from torchvision import io
import torch
import scipy.io as scio
import cv2
import numpy as np
from torchvision import utils
import torch.nn as nn
import math
import torch.nn.functional as F
from torchvision.transforms.functional import rotate, InterpolationMode, resize
import time
import matplotlib.pyplot as plt  # 导入 Matplotlib 工具包
from scipy.interpolate import interp1d  # 导入 scipy 中的一维插值工具 interp1d
from torch.utils.data import Dataset
import copy
import torch.distributions as tdist


class Simulation:
    def __init__(self):
        self.dataset = None
        self.dataloader = None
        self.psfmap = None
        self.device = None
        self.patch_length = None
        self.b_size = None
        self.fov_num = None
        self.enp_num = None
        self.psf_num = None
        self.psf_num_max = None
        self.img_size = None
        self.img_pixel_size = None
        self.wavelength = None
        self.flag_auto_calc = None
        self.is_special = None
        self.fov_hand_set = None
        self.img_index = None
        self.img_s = None
        self.psf = None
        self.view_pos = None
        self.now_epochs = None
        self.path = None
        self.loss_map_weight = None
        self.flag_simu_relative_illumination = None
        self.relative_illumination_map = None
        self.flag_simu_distortion = None
        self.distortion_map = None
        self.undistortion_map = None
        self.wave_path = None
        self.wave_sample_num = None
        self.noise_std = None
        self.disturbance = 0
        self.use_isp = None
        self.isp = None

    def sample_wave(self, flag_plot=False):
        path = self.wave_path
        sample_num = self.wave_sample_num
        raw_data = scio.loadmat(path)['wave']
        name = ['R', 'G', 'B']
        wavelength = {
            "R": None,
            "G": None,
            "B": None,
        }
        for i in range(3):
            wave_data = raw_data[0, i]
            fx = interp1d(wave_data[0], wave_data[1], kind='cubic')  # 由已知数据 (x,y) 求出插值函数 fx
            # 由插值函数 fx 计算插值点的函数值
            wave_main = wave_data[0, wave_data[1].argmax()]

            f = 0
            b = 0
            while wave_data[1, wave_data[1].argmax()-f:wave_data[1].argmax()+b].sum() < wave_data[1].sum()*0.95:
                if wave_data[1].argmax()-f > 0:
                    f = f + 1
                if wave_data[1].argmax()+b < len(wave_data[1])-1:
                    b = b + 1

            xInterp = np.append(np.linspace(wave_data[0, wave_data[1].argmax()-f], wave_main, sample_num // 2, endpoint=False), np.linspace(wave_main, wave_data[0, wave_data[1].argmax()+b], sample_num // 2 + 1))
            yInterp = fx(xInterp)  # 调用插值函数 fx，计算 xInterp 的函数值
            yInterp = yInterp/yInterp.sum()
            wavelength[name[i]] = (xInterp.tolist(), yInterp.tolist())
            # 绘图
            if flag_plot:
                plt.plot(xInterp, yInterp)
                plt.title(name[i])
                plt.show()

        return wavelength

    def set_path(self, input_root, is_special=False):
        # self.dataset = SimuDataset(input_root, is_special=is_special)
        # 所有图片的绝对路径
        img_s = os.listdir(input_root)
        self.is_special = is_special
        list1 = []
        index = 0
        for i in img_s:
            if i == 'resize0018.png' and is_special:
                self.img_index = index
            list1.append(os.path.join(input_root, i))
            index += 1
        self.img_s = list1

    def calc_relative_illumination_map(self, relative_factors, fov_pos, save=True, demo_root=r'./autodiff_demo/'):
        with torch.no_grad():
            device = self.device
            N, C, H, W = self.b_size, 3, self.img_size[0], self.img_size[1]
            interval_num = int(((H / 2) ** 2 + (W / 2) ** 2) ** 0.5) + 1
            illumination_list = F.interpolate(relative_factors.unsqueeze(0).unsqueeze(0),
                                              size=interval_num, mode='linear',
                                              align_corners=True).squeeze(0).squeeze(0)
            x, y = torch.meshgrid(
                torch.linspace(-H // 2, H // 2, H, device=device),
                torch.linspace(-W // 2, W // 2, W, device=device),
                indexing='ij',
            )
            illumination_map = illumination_list[torch.sqrt(torch.square(x) + torch.square(y)).long()]
            illumination_map = illumination_map / illumination_map.max()
            if save:
                utils.save_image(illumination_map, demo_root + '{iter}_relative_illumination_map.png'.format(iter=self.now_epochs))
            return illumination_map

    @staticmethod
    def calc_distortion_map(psf_cen_all, fov_pos):
        device = psf_cen_all.device
        psf_cen_mean = psf_cen_all[:, psf_cen_all.size(1)//2].square().sum(dim=-1).sqrt()
        fov_pos_real = psf_cen_mean / (psf_cen_mean.diff() / torch.Tensor(fov_pos).to(device).diff()).max()
        fx = interp1d(fov_pos, fov_pos_real.detach().cpu().numpy(), kind='cubic')
        return fx

    @staticmethod
    def calc_undistortion_map(psf_cen_all, fov_pos):
        device = psf_cen_all.device
        psf_cen_mean = psf_cen_all[:, psf_cen_all.size(1) // 2].square().sum(dim=-1).sqrt()
        fov_pos_real = psf_cen_mean / (psf_cen_mean.diff() / torch.Tensor(fov_pos).to(device).diff()).max()
        fx = interp1d(fov_pos_real.detach().cpu().numpy(), fov_pos, kind='cubic')
        return fx

    @staticmethod
    def tran_distort(blur_img, distort_func):
        N, C, H, W = blur_img.size()
        device = blur_img.device

        grid = torch.empty(N, H, W, 2).to(device)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing='ij',
        )

        r_pos = (x.square() + y.square()).sqrt()
        r_pos = r_pos / r_pos.max()
        item_t = torch.Tensor(distort_func(r_pos.detach().cpu().numpy())).to(device)
        item_t = item_t / item_t.max()
        distort_map = r_pos / item_t
        empty_map = ((x * distort_map).abs() > 1) | ((y * distort_map).abs() > 1)
        x = torch.clamp(x * distort_map, -1, 1)
        y = torch.clamp(y * distort_map, -1, 1)

        grid[..., 0] = x
        grid[..., 1] = y
        output = torch.grid_sampler(blur_img, grid, 0, 0, False)
        output[:, :, empty_map] = 0
        return output

    def tran_undistort(self):
        for index in range(len(self.img_s)):
            img_path = self.img_s[index]
            img = (io.read_image(img_path).float() / 255).unsqueeze(0).to(self.device)
            N, C, H, W = img.size()
            device = img.device
            grid = torch.empty(N, H, W, 2).to(device)
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, H, device=device),
                torch.linspace(-1, 1, W, device=device),
                indexing='ij',
            )
            r_pos = (x.square() + y.square()).sqrt()
            r_pos = r_pos / r_pos.max() * self.undistortion_map.x.max()
            item_t = torch.Tensor(self.undistortion_map(r_pos.detach().cpu().numpy())).to(device)
            distort_map = r_pos / item_t / r_pos.max()
            empty_map = ((x * distort_map).abs() > 1) | ((y * distort_map).abs() > 1)
            x = torch.clamp(x * distort_map, -1, 1)
            y = torch.clamp(y * distort_map, -1, 1)

            grid[..., 0] = x
            grid[..., 1] = y
            output = torch.grid_sampler(img, grid, 0, 0, False)
            output[:, :, empty_map] = 1e-4
            path = img_path[:-4] + 'undistort.png'
            utils.save_image(output, path)
            print(index)

    def simulation_to_psfmap(self, psf_all, view_pos, path, fast=False):

        self.view_pos = view_pos
        self.path = path
        self.psfmap = []
        device = self.device
        h, w = self.img_size[0], self.img_size[1]
        patch_length = self.patch_length
        h_num = round(h / patch_length)
        w_num = round(w / patch_length)
        index_h, index_w = torch.meshgrid(
            torch.linspace(0, h_num - 1, h_num, device=device),
            torch.linspace(0, w_num - 1, w_num, device=device),
            indexing='ij',
        )
        d_max = np.sqrt(h_num ** 2 + w_num ** 2) / 2
        d_loc = ((index_h + 0.5 - h_num / 2) ** 2 + (index_w + 0.5 - w_num / 2) ** 2).sqrt()
        standard_vector = torch.Tensor([0, 1]).to(device)
        yloc = index_h + 0.5 - h_num / 2
        xloc = index_w + 0.5 - w_num / 2
        location_vector = torch.stack((xloc, yloc), dim=2)
        norm_l = torch.stack((xloc, yloc), dim=2).square().sum(dim=2).sqrt()
        cos_angle = (location_vector * standard_vector).sum(dim=2) / norm_l
        rotate_theta = torch.arccos(cos_angle) / math.pi * 180
        rotate_theta[(xloc == 0) * (yloc == 0)] = 0
        rotate_theta[xloc > 0] = -rotate_theta[xloc > 0]
        rotate_theta = rotate_theta.reshape(h_num * w_num) + 180
        index = (torch.from_numpy(view_pos).unsqueeze(0).unsqueeze(0)).float().to(device) - (d_loc / d_max).unsqueeze(2)
        index = (index ** (-2)).abs()

        for psf_order in range(len(psf_all)):
            psf_single = self.psf = psf_all[psf_order]
            if fast:
                psf_used = ((index.unsqueeze(3).unsqueeze(3).unsqueeze(3) * psf_single.unsqueeze(0).unsqueeze(0)).sum(dim=2).squeeze(
                    2)).reshape(h_num * w_num, psf_single.size(1), psf_single.size(2), psf_single.size(3))
            else:
                psf_used = torch.zeros(h_num * w_num, psf_single.size(1), psf_single.size(2), psf_single.size(3)).to(device)
                for index_h in range(0, h_num):
                    for index_w in range(0, w_num):
                        psf_used[index_w + index_h * w_num] = (psf_single.transpose(0, 3)*index[index_h, index_w]).transpose(0, 3).sum(dim=0)
            # utils_ga.save_image(psf_used, './autodiff_demo/all_psf.png', nrow=15, normalize=True, scale_each=True, padding=1, pad_value=1)
            psf_used1 = self.my_rotate(psf_used, interpolation=InterpolationMode.BILINEAR, angle=rotate_theta)
            psf_used2 = psf_used1 / psf_used1.sum(dim=3).sum(dim=2).unsqueeze(2).unsqueeze(2)
            # utils_ga.save_image(psf_used2, './autodiff_demo/psf.png', nrow=w_num, normalize=True, scale_each=True)
            PSF_draw = psf_used2.reshape(h_num * w_num * psf_single.size(1), 1, psf_single.size(2), psf_single.size(3))
            save_filename = "iter" + str(self.now_epochs) + '_' + str(psf_order) + '.png'

            with torch.no_grad():
                PSF_save = PSF_draw.reshape(h_num * w_num, 3, PSF_draw.size(2), PSF_draw.size(3))
                item = PSF_save.reshape(PSF_save.size(0), -1).T
                item = (item / item.max(dim=0).values).T
                utils.save_image(item.reshape(h_num * w_num, 3, PSF_draw.size(2), PSF_draw.size(3)), os.path.join(self.path.demo_root, 'all_psf_' + save_filename), nrow=w_num
                                 , sacle_each=True, normalize=True, padding=1, pad_value=1)

            self.psfmap.append(PSF_draw)

    def psfmap_to_img(self, simulation):
        self.use_isp = simulation.use_isp
        self.psfmap = simulation.psfmap
        self.device = simulation.device
        self.patch_length = simulation.patch_length
        self.b_size = simulation.b_size
        self.img_s = simulation.img_s
        self.img_size = simulation.img_size
        self.path = simulation.path
        self.psf = simulation.psf
        self.loss_map_weight = None
        self.flag_simu_relative_illumination = simulation.flag_simu_relative_illumination
        self.relative_illumination_map = simulation.relative_illumination_map
        self.flag_simu_distortion = simulation.flag_simu_distortion
        self.distortion_map = simulation.distortion_map
        self.wave_path = None
        self.wave_sample_num = None
        self.noise_std = simulation.noise_std
        path = self.path
        device = self.device
        psf = self.psf
        patch_length = self.patch_length

        for index in range(len(self.img_s)):
            index = 1
            self.isp = ISP(simulation.device, disturb=0.05)
            img_path = self.img_s[index]
            img = (io.read_image(img_path).float() / 255).unsqueeze(0).to(self.device)[:, :3]

            if self.flag_simu_relative_illumination:
                img = img * self.relative_illumination_map
            if self.flag_simu_distortion:
                save_filename = os.path.basename(img_path)
                clear_path = path.clear_root + save_filename[:-4] + '.png'
                clear_img = self.tran_distort(img, self.distortion_map)
                utils.save_image(clear_img, clear_path)
            # img = img / img.mean() * 0.3508
            img[img > 0.1] = 0.45
            img[img < 0.1] = 0.18

            if self.use_isp:
                img = self.isp.backward(img)
            N, C, H, W = img.shape
            # 计算图像真实尺寸
            pad_whole = (psf.size(3) - 1) // 2
            pad = (psf.size(3) - 1) // 2
            h, w = self.img_size[0], self.img_size[1]
            h_num = round(h / patch_length)
            w_num = round(w / patch_length)
            img_pad = F.pad(img, (pad_whole, pad_whole, pad_whole, pad_whole), mode='reflect').to(device)
            inputs_pad = F.unfold(img_pad, patch_length + 2 * pad, 1, 0, patch_length).transpose(1, 2).reshape(N, C * h_num * w_num,
                                                                                                               patch_length + 2 * pad,
                                                                                                               patch_length + 2 * pad)
            for psf_order in range(len(self.psfmap)):

                outputs = F.conv2d(inputs_pad, self.psfmap[psf_order], stride=1, groups=C * h_num * w_num)

                blur_img = F.fold(
                    outputs.reshape(N, h_num * w_num, psf.size(1) * patch_length * patch_length).transpose(1, 2),
                    (self.img_size[0], self.img_size[1]), patch_length, 1, 0, patch_length)

                if self.flag_simu_distortion:
                    blur_img = self.tran_distort(blur_img, self.distortion_map)

                if self.flag_simu_relative_illumination:
                    blur_img = blur_img * self.relative_illumination_map

                if self.use_isp:
                    blur_img = self.isp.forward(blur_img)

                # if self.noise_std is not None:
                #     noise = torch.normal(mean=torch.zeros_like(blur_img[:, 0:1, :, :]), std=self.noise_std)
                #     # if index == 0:
                #     #     save_filename = 'noise' + os.path.basename(img_path)
                #     #     blur_path = os.path.join(path.blur_root, save_filename)
                #     #     utils_ga.save_image(noise, blur_path)
                #     blur_img = blur_img + noise
                #     blur_img = torch.clip(blur_img, 0, 1)

                # save_filename = os.path.basename(img_path)
                # blur_path = os.path.join(path.blur_root, save_filename)
                # utils_ga.save_image(blur_img, blur_path)
                save_filename = os.path.basename(img_path)
                blur_path = path.blur_root + save_filename[:-4] + '_' + str(psf_order) + '.png'
                utils.save_image(blur_img, blur_path)
                # utils_ga.save_image(clear_img[i], clear_path)
                print('{num}'.format(num=index))
                # 存储模糊图像

    def simulation_to_img(self, psf, view_pos, path, now_epochs=-1, flag_simulation_random=False, specific_index=None, flag_simulation_all=False):
        self.isp = ISP(self.device, disturb=0.0)
        self.view_pos = view_pos
        self.path = path
        index = None
        clear_img_s = []
        blur_img_s = []
        filename_s = []
        if specific_index is not None:
            index = specific_index
        if flag_simulation_random:
            index = np.random.randint(0, len(self.img_s), size=self.b_size)
        if index is not None:
            for psf_order in range(len(psf)):

                self.psf = psf[psf_order]
                clear_img, blur_img, filename = self.getitem(index, fast=True)

                # if self.flag_simu_relative_illumination:
                #     blur_img = blur_img * self.relative_illumination_map
                # if self.noise_std is not None:
                #     noise = torch.normal(mean=torch.zeros_like(blur_img), std=self.noise_std)
                #     blur_img = blur_img + noise
                #     blur_img = torch.clip(blur_img, 0, 1)

                clear_img_s.append(clear_img)
                blur_img_s.append(blur_img)
                filename_s.append(filename)
            return clear_img_s, blur_img_s, filename_s

        else:
            self.psf = psf[len(psf) // 2]
            for index in range(len(self.img_s)):
                # PSF分块卷积源图像，得到仿真的模糊图像，随机选择b_size张图片分块卷积
                if index == 0:
                    save_psf_all = True
                else:
                    save_psf_all = False
                clear_img, blur_img, filename_s = self.getitem([index], fast=True, save_psf_all=save_psf_all)
                if self.flag_simu_relative_illumination:
                    blur_img = blur_img * self.relative_illumination_map
                if self.noise_std is not None:
                    noise = torch.normal(mean=torch.zeros_like(blur_img), std=self.noise_std)
                    # if index == 0:
                    #     save_filename = 'noise' + os.path.basename(img_path)
                    #     blur_path = os.path.join(path.blur_root, save_filename)
                    #     utils_ga.save_image(noise, blur_path)
                    blur_img = blur_img + noise
                    blur_img = torch.clip(blur_img, 0, 1)
                # 存储模糊图像和清晰图像
                for i in range(self.b_size):
                    save_filename = filename_s[i]
                    # clear_path = os.path.join(clear_root, 'clear' + save_filename)
                    blur_path = os.path.join(path.blur_root, save_filename)
                    utils.save_image(blur_img[i], blur_path)
                    # utils_ga.save_image(clear_img[i], clear_path)
                print('iteration:{num}'.format(num=index))

    def getitem(self, index, fast=True, save_psf_all=False):
        patch_length = self.patch_length
        PSF = self.psf
        if self.is_special:
            index[-1] = self.img_index
        h, w = self.img_size[0], self.img_size[1]
        clear_img_s = torch.zeros(len(index), 3, h, w).to(self.device)
        file_name_s = []
        # time1 = time.time()
        for i in range(len(index)):
            img_path = self.img_s[index[i]]
            img = io.read_image(img_path).float() / 255
            clear_img_s[i] = img[:3]
            file_name_s.append(os.path.basename(img_path))

        if self.use_isp:
            isp_img_s = self.isp.backward(clear_img_s)
        else:
            isp_img_s = clear_img_s
        # time2 = time.time()
        # 定义patch_conv相关参数
        h_num = round(h / patch_length)
        w_num = round(w / patch_length)
        if fast:
            blur_img_s, PSF_draw = self.super_fast_patch_wise_conv(PSF, isp_img_s, h_num=h_num, w_num=w_num)
        else:
            blur_img_s, PSF_draw = self.fast_patch_wise_conv(PSF, isp_img_s, h_num=h_num, w_num=w_num)

        if self.use_isp:
            blur_img_s = self.isp.forward(blur_img_s)

        # time3 = time.time()
        if save_psf_all:
            save_filename = "iter" + str(self.now_epochs) + '.png'
            utils.save_image(PSF_draw.reshape(h_num*w_num, 3, PSF_draw.size(2), PSF_draw.size(3)), os.path.join(self.path.demo_root, 'all_psf_'+save_filename), nrow=w_num,
                             normalize=True, sacle_each=True)
        # time4 = time.time()
        # print(time2 - time1)
        # print(time3 - time2)
        # print(time4 - time3)
        # utils.save_image(clear_img_s, 'clear.png')
        # utils.save_image(isp_img_s, 'isp.png')
        # utils.save_image(blur_img_s, 'blur.png')
        return clear_img_s, blur_img_s, file_name_s

    def fast_patch_wise_conv(self, psf, img, h_num=100, w_num=100):
        patch_length = self.patch_length
        view_pos = self.view_pos
        N, C, H, W = img.shape
        # 计算图像真实尺寸
        device = self.device
        pad_whole = (psf.size(3) - 1) // 2
        pad = (psf.size(3) - 1) // 2
        img_pad = F.pad(img, (pad_whole, pad_whole, pad_whole, pad_whole), mode='reflect').to(device)
        # h_index = np.linspace(0, h_num-1, h_num)
        # w_index = np.linspace(0, w_num-1, w_num)
        inputs_pad = torch.zeros(N, C * h_num * w_num, patch_length + 2 * pad, patch_length + 2 * pad).to(device)
        PSF_draw = torch.zeros(h_num * w_num * psf.size(1), 1, psf.size(2), psf.size(3)).to(device)
        blur_img = torch.zeros_like(img).to(device)
        time1 = time.time()
        for index_h in range(0, h_num):
            for index_w in range(0, w_num):
                # FOV判断
                d_max = np.sqrt(h_num ** 2 + w_num ** 2) / 2
                d_loc = np.sqrt((index_h + 0.5 - h_num / 2) ** 2 + (index_w + 0.5 - w_num / 2) ** 2)
                if d_loc / d_max >= 1:
                    psf_used = psf[-1].unsqueeze(0)
                elif d_loc / d_max <= 0:
                    psf_used = psf[0].unsqueeze(0)
                else:
                    index_up = np.where(view_pos > d_loc / d_max)[0][0]
                    index_down = index_up - 1
                    item1 = view_pos[index_up] - d_loc / d_max
                    item2 = d_loc / d_max - view_pos[index_down]
                    a = item1 + item2
                    item1 = item1 / a
                    item2 = item2 / a
                    psf_used = (psf[index_up] / psf[index_up].sum() * item2 + psf[index_down] / psf[
                        index_down].sum() * item1).unsqueeze(0)

                # 确定旋转角并旋转图像
                standard_vector = np.array([0, 1])
                yloc = index_h + 0.5 - h_num / 2
                xloc = index_w + 0.5 - w_num / 2
                location_vector = np.array([xloc + 1e-8, yloc + 1e-8])
                norm_s = np.sqrt(standard_vector.dot(standard_vector))
                norm_l = np.sqrt(location_vector.dot(location_vector))
                cos_angle = standard_vector.dot(location_vector) / (norm_s * norm_l)
                rotate_theta = np.degrees(np.arccos(cos_angle))
                if xloc == 0 and yloc == 0:
                    rotate_theta = 0
                if xloc < 0:
                    rotate_theta = -rotate_theta
                # 旋转
                psf_used1 = rotate(psf_used, interpolation=InterpolationMode.BILINEAR, angle=rotate_theta)
                psf_used2 = torch.stack(
                    (psf_used1[:, 0] / torch.sum(psf_used1[:, 0]),
                     psf_used1[:, 1] / torch.sum(psf_used1[:, 1]),
                     psf_used1[:, 2] / torch.sum(psf_used1[:, 2])), dim=1)
                # 可视化
                PSF_draw[(index_w + index_h * w_num) * psf.size(1):(index_w + index_h * w_num + 1) * psf.size(
                    1)] = psf_used2.transpose(0, 1)
                inputs_pad[:, (index_w + index_h * w_num) * C:(index_w + index_h * w_num + 1) * C] = \
                    img_pad[:, :, index_h * patch_length + pad_whole - pad:(index_h + 1) * patch_length + pad_whole + pad,
                            index_w * patch_length + pad_whole - pad:(index_w + 1) * patch_length + pad_whole + pad]

        outputs = F.conv2d(inputs_pad, PSF_draw, stride=1, groups=C * h_num * w_num)

        for index_h in range(0, h_num):
            for index_w in range(0, w_num):
                blur_img[:, :, index_h * patch_length:(index_h + 1) * patch_length,
                        index_w * patch_length:(index_w + 1) * patch_length] = outputs[:, (index_w + index_h * w_num) * C:(index_w + index_h * w_num + 1) * C]

        # for i in range(N):
        #     blur_img[i] = utils_ga.make_grid(outputs[i].reshape(h_num * w_num, C, patch_length, patch_length), padding=0, nrow=w_num)
        time3 = time.time()
        print('patch: {time}'.format(time=patch_length))
        print('fast_conv_time: {time}'.format(time=time3 - time1))

        return blur_img, PSF_draw

    def super_fast_patch_wise_conv(self, psf, img, h_num=100, w_num=100):
        patch_length = self.patch_length
        view_pos = self.view_pos

        N, C, H, W = img.shape
        # 计算图像真实尺寸
        device = self.device
        pad_whole = (psf.size(3) - 1) // 2
        pad = (psf.size(3) - 1) // 2

        img_pad = F.pad(img, (pad_whole, pad_whole, pad_whole, pad_whole), mode='reflect').to(device)
        inputs_pad = F.unfold(img_pad, patch_length + 2 * pad, 1, 0, patch_length).transpose(1, 2).reshape(N,
                                                                                                           C * h_num * w_num,
                                                                                                           patch_length + 2 * pad,
                                                                                                           patch_length + 2 * pad)
        # utils_ga.save_image(inputs_pad.reshape(h_num * w_num, C, patch_length + 2 * pad, patch_length + 2 * pad),
        #                  './autodiff_demo/psf_tensor.png', nrow=w_num, normalize=True, padding=3, pad_value=1)
        # 解除for循环
        index_h, index_w = torch.meshgrid(
            torch.linspace(0, h_num - 1, h_num, device=device),
            torch.linspace(0, w_num - 1, w_num, device=device),
            indexing='ij',
        )
        d_max = np.sqrt(h_num ** 2 + w_num ** 2) / 2
        d_loc = ((index_h + 0.5 - h_num / 2) ** 2 + (index_w + 0.5 - w_num / 2) ** 2).sqrt()
        standard_vector = torch.Tensor([0, 1]).to(device)
        yloc = index_h + 0.5 - h_num / 2
        xloc = index_w + 0.5 - w_num / 2
        location_vector = torch.stack((xloc, yloc), dim=2)
        norm_l = torch.stack((xloc, yloc), dim=2).square().sum(dim=2).sqrt()
        cos_angle = (location_vector * standard_vector).sum(dim=2) / norm_l
        rotate_theta = torch.arccos(cos_angle) / math.pi * 180
        rotate_theta[(xloc == 0) * (yloc == 0)] = 0
        rotate_theta[xloc > 0] = -rotate_theta[xloc > 0]
        rotate_theta = rotate_theta.reshape(h_num * w_num)
        index = (torch.from_numpy(view_pos).unsqueeze(0).unsqueeze(0)).float().to(device) - (d_loc / d_max).unsqueeze(2)
        index = index ** (-2)
        psf_used = ((index.unsqueeze(3).unsqueeze(3).unsqueeze(3) * psf.unsqueeze(0).unsqueeze(0)).sum(dim=2).squeeze(
            2)).reshape(h_num * w_num, psf.size(1), psf.size(2), psf.size(3))
        # utils_ga.save_image(psf_used, './autodiff_demo/all_psf.png', nrow=15, normalize=True, scale_each=True, padding=1, pad_value=1)
        time1 = time.time()
        psf_used1 = self.my_rotate(psf_used, interpolation=InterpolationMode.BILINEAR, angle=rotate_theta)
        # time2 = time.time()
        psf_used2 = psf_used1 / psf_used1.sum(dim=3).sum(dim=2).unsqueeze(2).unsqueeze(2)
        # utils_ga.save_image(psf_used2, './autodiff_demo/psf.png', nrow=w_num, normalize=True, scale_each=True)
        PSF_draw = psf_used2.reshape(h_num * w_num * psf.size(1), 1, psf.size(2), psf.size(3))
        outputs = F.conv2d(inputs_pad, PSF_draw, stride=1, groups=C * h_num * w_num)

        blur_img = F.fold(outputs.reshape(N, h_num * w_num, psf.size(1) * patch_length * patch_length).transpose(1, 2),
                          (self.img_size[0], self.img_size[1]), patch_length, 1, 0, patch_length)
        # for i in range(N):
        #     blur_img[i] = utils_ga.make_grid(outputs[i].reshape(h_num * w_num, C, patch_length, patch_length), padding=0, nrow=w_num)
        # time3 = time.time()
        # print('patch: {time}'.format(time=patch_length))
        # print('super_fast_pre_time: {time}'.format(time=time2 - time1))
        # print('super_fast_conv_time: {time}'.format(time=time3 - time2))

        return blur_img, PSF_draw

    @staticmethod
    def my_rotate(img, interpolation=InterpolationMode.BILINEAR, angle=None):

        translate = [0.0, 0.0]
        scale = 1.0
        shear = [0, 0]
        center = [0.0, 0.0]
        rot = angle / 180 * math.pi
        sx, sy = [math.radians(s) for s in shear]
        cx, cy = center
        tx, ty = translate
        # RSS without scaling
        a = torch.cos(rot - sy) / math.cos(sy)
        b = -torch.cos(rot - sy) * math.tan(sx) / math.cos(sy) - torch.sin(rot)
        c = torch.sin(rot - sy) / math.cos(sy)
        d = -torch.sin(rot - sy) * math.tan(sx) / math.cos(sy) + torch.cos(rot)
        zeros = torch.zeros_like(a)
        # Inverted rotation matrix with scale and shear
        # det([[a, b], [c, d]]) == 1, since det(rotation) = 1 and det(shear) = 1
        matrix = torch.stack((d, -b, zeros, -c, a, zeros), dim=1)
        n, c, w, h = img.shape[0], img.shape[1], img.shape[3], img.shape[2]
        ow, oh = (w, h)
        theta = matrix.reshape(-1, 2, 3)
        # grid = _gen_affine_grid(theta, w=w, h=h, ow=ow, oh=oh)
        d = 0.5

        base_grid = torch.empty(n, oh, ow, 3, dtype=theta.dtype, device=theta.device)
        x_grid = torch.linspace(-ow * 0.5 + d, ow * 0.5 + d - 1, steps=ow, device=theta.device)
        base_grid[..., 0].copy_(x_grid)
        y_grid = torch.linspace(-oh * 0.5 + d, oh * 0.5 + d - 1, steps=oh, device=theta.device).unsqueeze_(-1)
        base_grid[..., 1].copy_(y_grid)
        base_grid[..., 2].fill_(1)
        rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype, device=theta.device)

        # time2 = time.time()
        item = base_grid.view(n, oh * ow, 3)
        # time3 = time.time()

        output_grid = item.bmm(rescaled_theta)
        # output_grid = torch.rand(600, 2601, 3).to(torch.device('cuda:0')).bmm(torch.rand(600, 3, 2).to(torch.device('cuda:0')))

        # time4 = time.time()
        grid = output_grid.view(n, oh, ow, 2)
        # _apply_grid_transform(img, grid, interpolation, fill=fill)

        output = torch.grid_sampler(img, grid, 0, 0, False)

        # print(time4 - time3)
        # print(time3 - time2)
        return output
        # img = grid_sample(img, grid, mode=interpolation.value, padding_mode="zeros", align_corners=False)


class ISP:
    def __init__(self, device, disturb=0.):
        self.device = device
        self.disturb = disturb
        self.blc = torch.Tensor([0.0])[0].to(device)
        self.gamma = 2.3 * (1 + (torch.rand(1)[0] - 0.5) / 0.5 * disturb).to(device)
        wb = torch.Tensor([2.8429,  1.8855]).to(device) * (1 + (torch.rand(2) - 0.5) / 0.5 * disturb).to(device)
        ccm = torch.Tensor([2.07, -0.99, -0.19, 1.59, 0.09, -0.55]).to(device) * (1 + (torch.rand(6) - 0.5) / 0.5 * disturb).to(device)
        self.wb = torch.Tensor([wb[0], 1, wb[1]]).to(device)
        self.ccm = torch.Tensor([[ccm[0], ccm[1], 1 - ccm[0] - ccm[1]], [ccm[2], ccm[3], 1 - ccm[2] - ccm[3]],
                                 [ccm[4], ccm[5], 1 - ccm[4] - ccm[5]]]).to(device)

    def forward(self, xx):
        n, c, h, w = xx.size()
        # 马赛克
        item = torch.zeros(n, 4, h // 2, w // 2).to(self.device)
        item[:, 0] = xx[:, 0:1, 0::2, 0::2]
        item[:, 1] = xx[:, 1:2, 0::2][:, :, :, 1::2]
        item[:, 2] = xx[:, 1:2, 1::2][:, :, :, 0::2]
        item[:, 3] = xx[:, 2:, 1::2, 1::2]

        # 加噪声
        item = self.add_noise(torch.clamp(item, min=1e-8))
        # utils.save_image(item[:, 2] * 3, r'./checkerboard_simu/MOS_S1/gg2.png')
        # utils.save_image(item[:, 3] * 3, r'./checkerboard_simu/MOS_S1/bb.png')
        # utils.save_image(item[:, 1] * 3, r'./checkerboard_simu/MOS_S1/gg1.png')
        # utils.save_image(item[:, 0] * 3, r'./checkerboard_simu/MOS_S1/rr.png')
        item = torch.clamp((item - self.blc) / (1 - self.blc), min=0)
        # utils.save_image(torch.cat((item[:, 0], item[:, 1], item[:, 3]), dim=0), 'simu.png')

        x = torch.zeros(n, 1, h, w).to(self.device)

        # x[:, :, 1::2, 1::2] = (io.read_image('b.png')[0:1].float() / 255).to(self.device).unsqueeze(0)
        # x[:, :, 0::2, 0::2] = (io.read_image('r.png')[0:1].float() / 255).to(self.device).unsqueeze(0)
        # x[:, :, 0::2][:, :, :, 1::2] = (io.read_image('g1.png')[0:1].float() / 255).to(self.device).unsqueeze(0)
        # x[:, :, 1::2][:, :, :, 0::2] = (io.read_image('g2.png')[0:1].float() / 255).to(self.device).unsqueeze(0)

        x[:, :, 1::2, 1::2] = item[:, 3] * self.wb[2]
        x[:, :, 0::2, 0::2] = item[:, 0] * self.wb[0]
        x[:, :, 0::2][:, :, :, 1::2] = item[:, 1]
        x[:, :, 1::2][:, :, :, 0::2] = item[:, 2]

        x = torch.clamp(self.demosaic_bilinear(x), min=0, max=1)

        x = x.reshape(n, c, -1).transpose(0, 2).transpose(1, 2)
        # x = x * self.wb
        x = self.ccm.mm(x.T.reshape(c, -1))
        x = torch.clamp(x, min=1e-8) ** (1 / self.gamma)
        return x.reshape(c, n, -1).transpose(0, 1).reshape(n, c, h, w)

    def backward(self, x):
        n, c, h, w = x.size()
        x = x ** self.gamma
        x = x.reshape(n, c, -1).transpose(0, 1)
        x = (self.ccm.inverse()).mm(x.reshape(c, -1))
        x = (x.T / self.wb).T
        x = x * (1 - self.blc) + self.blc
        return x.reshape(c, n, -1).transpose(0, 1).reshape(n, c, h, w)

    def demosaic_bilinear(self, x):
        device = self.device
        n, _, h, w = x.size()
        # 定义马赛克索引, 012对应RGB
        pattern = torch.zeros_like(x)
        pattern[:, :, 1::2, 1::2] = 2
        pattern[:, :, 0::2][:, :, :, 1::2] = 1
        pattern[:, :, 1::2][:, :, :, 0::2] = 1
        y = torch.zeros(n, 3, h, w).to(x.device)
        y[:, 0:1, :, :][pattern == 0] = x[pattern == 0]
        y[:, 1:2, :, :][pattern == 1] = x[pattern == 1]
        y[:, 2:, :, :][pattern == 2] = x[pattern == 2]
        y_pad = F.pad(y, (3, 3, 3, 3), 'reflect').to(device)
        pattern_pad = F.pad(pattern, (3, 3, 3, 3), 'constant', 999).to(device)
        # 给蓝色上的红色像素插值
        idx_34 = list(torch.where(pattern_pad == 2))
        idx_23 = copy.deepcopy(idx_34)
        idx_23[2] -= 1
        idx_23[3] -= 1
        idx_25 = copy.deepcopy(idx_34)
        idx_25[2] -= 1
        idx_25[3] += 1
        idx_43 = copy.deepcopy(idx_34)
        idx_43[2] += 1
        idx_43[3] -= 1
        idx_45 = copy.deepcopy(idx_34)
        idx_45[2] += 1
        idx_45[3] += 1
        r_23 = y_pad[:, 0:1, :, :][idx_23]
        r_25 = y_pad[:, 0:1, :, :][idx_25]
        r_43 = y_pad[:, 0:1, :, :][idx_43]
        r_45 = y_pad[:, 0:1, :, :][idx_45]
        y_pad[:, 0:1, :, :][idx_34] = (r_23 + r_25 + r_43 + r_45) / 4

        # 给红色上的蓝色像素插值
        idx_34 = list(torch.where(pattern_pad == 0))
        idx_23 = copy.deepcopy(idx_34)
        idx_23[2] -= 1
        idx_23[3] -= 1
        idx_25 = copy.deepcopy(idx_34)
        idx_25[2] -= 1
        idx_25[3] += 1
        idx_43 = copy.deepcopy(idx_34)
        idx_43[2] += 1
        idx_43[3] -= 1
        idx_45 = copy.deepcopy(idx_34)
        idx_45[2] += 1
        idx_45[3] += 1
        b_23 = y_pad[:, 2:, :, :][idx_23]
        b_25 = y_pad[:, 2:, :, :][idx_25]
        b_43 = y_pad[:, 2:, :, :][idx_43]
        b_45 = y_pad[:, 2:, :, :][idx_45]
        y_pad[:, 2:, :, :][idx_34] = (b_23 + b_25 + b_43 + b_45) / 4

        # 给绿色像素插值
        idx_34 = list(torch.where((pattern_pad != 1) & (pattern_pad != 999)))
        idx_33 = copy.deepcopy(idx_34)
        idx_33[3] -= 1
        idx_24 = copy.deepcopy(idx_34)
        idx_24[2] -= 1
        idx_35 = copy.deepcopy(idx_34)
        idx_35[3] += 1
        idx_44 = copy.deepcopy(idx_34)
        idx_44[2] += 1
        g_33 = y_pad[:, 1:2, :, :][idx_33]
        g_24 = y_pad[:, 1:2, :, :][idx_24]
        g_35 = y_pad[:, 1:2, :, :][idx_35]
        g_44 = y_pad[:, 1:2, :, :][idx_44]
        y_pad[:, 1:2, :, :][idx_34] = (g_33 + g_24 + g_35 + g_44) / 4

        # 剩下蓝色红色像素插值
        idx_33 = list(torch.where((pattern_pad == 1)))
        idx_32 = copy.deepcopy(idx_33)
        idx_32[3] -= 1
        idx_34 = copy.deepcopy(idx_33)
        idx_34[3] += 1
        idx_23 = copy.deepcopy(idx_33)
        idx_23[2] -= 1
        idx_43 = copy.deepcopy(idx_33)
        idx_43[2] += 1

        r_32 = y_pad[:, 0:1, :, :][idx_32]
        r_34 = y_pad[:, 0:1, :, :][idx_34]
        b_23 = y_pad[:, 2:, :, :][idx_23]
        b_43 = y_pad[:, 2:, :, :][idx_43]
        y_pad[:, 0:1, :, :][idx_33] = (r_32 + r_34) / 2
        y_pad[:, 2:, :, :][idx_33] = (b_23 + b_43) / 2

        return y_pad[:, :, 3:-3, 3:-3]

    def random_noise_levels(self):
        """Generates random noise levels from a log-log linear distribution."""
        log_min_shot_noise = np.log(0.0001)
        log_max_shot_noise = np.log(0.012)
        log_shot_noise = torch.FloatTensor(1).uniform_(log_min_shot_noise, log_max_shot_noise)
        shot_noise = torch.exp(log_shot_noise)
        line = lambda x: 2.18 * x + 1.20
        n = tdist.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([0.26]))
        log_read_noise = line(log_shot_noise) + n.sample()
        read_noise = torch.exp(log_read_noise)
        return shot_noise, read_noise

    def add_noise(self, image, shot_noise=0.0001, read_noise=0.0000005):
        """Adds random shot (proportional to image) and read (independent) noise."""
        image = image.transpose(1, 3).transpose(1, 2)
        variance = image * shot_noise + read_noise
        n = tdist.Normal(loc=torch.zeros_like(variance), scale=torch.sqrt(variance))
        noise = n.sample()
        out = image + noise
        return out.transpose(1, 3).transpose(2, 3)


class SimuDataset(Dataset):
    def __init__(self, input_root, is_special=False):
        img_s = os.listdir(input_root)
        self.is_special = is_special
        list1 = []
        index = 0
        for i in img_s:
            if i == 'resize0018.png' and is_special:
                self.img_index = index
            list1.append(os.path.join(input_root, i))
            index += 1
        self.img_s = list1

    # 根据索引获取data和label
    def __getitem__(self, index):
        file_name_s = []
        for i in range(len(index)):
            img_path = self.img_s[index[i]]
            img = io.read_image(img_path).float() / 255
            file_name_s.append(os.path.basename(img_path))
        return img  # 以元组的形式返回

    # 获取数据集的大小
    def __len__(self):
        return len(self.img_s)

