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


class Simulation:
    def __init__(self):
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
        self.iteration = None
        self.path = None
        self.loss_map_weight = None
        self.flag_simu_relative_illumination = None
        self.relative_illumination_map = None
        self.flag_simu_distortion = None
        self.wave_path = None
        self.wave_sample_num = None
        self.noise_std = None
        self.sol_per_pop = None

    def sample_wave(self, flag_plot=True):
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
                utils.save_image(illumination_map, demo_root + '{iter}_relative_illumination_map.png'.format(iter=self.iteration))
            return illumination_map

    def simulation_to_psfmap(self, psf, view_pos, path, fast=False):
        self.psf = psf
        self.view_pos = view_pos
        self.path = path
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
        rotate_theta = rotate_theta.reshape(h_num * w_num)
        index = (torch.from_numpy(view_pos).unsqueeze(0).unsqueeze(0)).float().to(device) - (d_loc / d_max).unsqueeze(2)
        index = index ** (-4)
        if fast:
            psf_used = ((index.unsqueeze(3).unsqueeze(3).unsqueeze(3) * psf.unsqueeze(0).unsqueeze(0)).sum(dim=2).squeeze(
                2)).reshape(h_num * w_num, psf.size(1), psf.size(2), psf.size(3))
        else:
            psf_used = torch.zeros(h_num * w_num, psf.size(1), psf.size(2), psf.size(3)).to(device)
            for index_h in range(0, h_num):
                for index_w in range(0, w_num):
                    psf_used[index_w + index_h * w_num] = (psf.transpose(0, 3)*index[index_h, index_w]).transpose(0, 3).sum(dim=0)
        # utils_ga.save_image(psf_used, './autodiff_demo/all_psf.png', nrow=15, normalize=True, scale_each=True, padding=1, pad_value=1)
        psf_used1 = self.my_rotate(psf_used, interpolation=InterpolationMode.BILINEAR, angle=rotate_theta)
        psf_used2 = psf_used1 / psf_used1.sum(dim=3).sum(dim=2).unsqueeze(2).unsqueeze(2)
        # utils_ga.save_image(psf_used2, './autodiff_demo/psf.png', nrow=w_num, normalize=True, scale_each=True)
        PSF_draw = psf_used2.reshape(h_num * w_num * psf.size(1), 1, psf.size(2), psf.size(3))
        save_filename = "iter" + str(self.iteration) + '.png'
        utils.save_image(PSF_draw.reshape(h_num * w_num, 3, PSF_draw.size(2), PSF_draw.size(3)),
                         os.path.join(self.path.demo_root, 'all_psf_' + save_filename), nrow=w_num, normalize=True, sacle_each=True)
        self.psfmap = PSF_draw

    def psfmap_to_img(self, simulation):
        self.psfmap = simulation.psfmap
        self.device = simulation.device
        self.patch_length = simulation.patch_length
        self.b_size = simulation.b_size
        self.img_s = simulation.img_s
        self.img_size = simulation.img_size
        self.path = simulation.path
        self.psf = simulation.psf
        self.loss_map_weight = None
        self.flag_simu_relative_illumination = None
        self.relative_illumination_map = None
        self.flag_simu_distortion = None
        self.wave_path = None
        self.wave_sample_num = None
        self.noise_std = simulation.noise_std
        path = self.path
        device = self.device
        psf = self.psf
        patch_length = self.patch_length

        for index in range(len(self.img_s)):
            img_path = self.img_s[index]
            img = (io.read_image(img_path).float() / 255).unsqueeze(0)
            N, C, H, W = img.shape
            # 计算图像真实尺寸
            pad_whole = (psf.size(3) - 1) // 2
            pad = (psf.size(3) - 1) // 2
            h, w = self.img_size[0], self.img_size[1]
            h_num = round(h / patch_length)
            w_num = round(w / patch_length)
            img_pad = F.pad(img, (pad_whole, pad_whole, pad_whole, pad_whole), mode='reflect').to(device)
            inputs_pad = F.unfold(img_pad, patch_length + 2 * pad, 1, 0, patch_length).transpose(1, 2).reshape(N,
                                                                                                               C * h_num * w_num,
                                                                                                               patch_length + 2 * pad,
                                                                                                               patch_length + 2 * pad)
            outputs = F.conv2d(inputs_pad, self.psfmap, stride=1, groups=C * h_num * w_num)

            blur_img = F.fold(
                outputs.reshape(N, h_num * w_num, psf.size(1) * patch_length * patch_length).transpose(1, 2),
                (self.img_size[0], self.img_size[1]), patch_length, 1, 0, patch_length)
            if self.flag_simu_relative_illumination:
                blur_img = blur_img * self.relative_illumination_map
            # save_filename = os.path.basename(img_path)
            # blur_path = os.path.join(path.blur_root, save_filename)
            # utils_ga.save_image(blur_img, blur_path)
            # 模拟噪声
            if self.noise_std is not None:
                noise = torch.normal(mean=torch.zeros_like(blur_img), std=self.noise_std)
                # if index == 0:
                #     save_filename = 'noise' + os.path.basename(img_path)
                #     blur_path = os.path.join(path.blur_root, save_filename)
                #     utils_ga.save_image(noise, blur_path)
                blur_img = blur_img + noise
                blur_img = torch.clip(blur_img, 0, 1)
            # 存储模糊图像
            save_filename = os.path.basename(img_path)
            blur_path = os.path.join(path.blur_root, save_filename)
            utils.save_image(blur_img, blur_path)
            # utils_ga.save_image(clear_img[i], clear_path)
            print('{num}'.format(num=index))

    def simulation_to_img(self, psf, view_pos, path, iteration=-1, flag_simulation_random=False, specific_index=None):
        self.psf = psf
        self.view_pos = view_pos
        self.iteration = iteration
        self.path = path
        if specific_index is not None:
            clear_img, blur_img, filename_s = self.getitem(specific_index, fast=True)
            if self.flag_simu_relative_illumination:
                blur_img = blur_img * self.relative_illumination_map
            return clear_img, blur_img, filename_s
        if flag_simulation_random:
            index = np.random.randint(0, len(self.img_s), size=self.b_size)
            clear_img, blur_img, filename_s = self.getitem(index, fast=True)
            if self.flag_simu_relative_illumination:
                blur_img = blur_img * self.relative_illumination_map
            return clear_img, blur_img, filename_s
        else:
            for index in range(len(self.img_s)):
                # PSF分块卷积源图像，得到仿真的模糊图像，随机选择b_size张图片分块卷积
                if index == 0:
                    save_psf_all = True
                else:
                    save_psf_all = False
                clear_img, blur_img, filename_s = self.getitem([index], fast=True, save_psf_all=save_psf_all)
                if self.flag_simu_relative_illumination:
                    blur_img = blur_img * self.relative_illumination_map
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
        for i in range(len(index)):
            img_path = self.img_s[index[i]]
            img = io.read_image(img_path).float() / 255
            clear_img_s[i] = img
            file_name_s.append(os.path.basename(img_path))
        # 定义patch_conv相关参数
        h_num = round(h / patch_length)
        w_num = round(w / patch_length)
        if fast:
            blur_img_s, PSF_draw = self.super_fast_patch_wise_conv(PSF, clear_img_s, h_num=h_num, w_num=w_num)
        else:
            blur_img_s, PSF_draw = self.fast_patch_wise_conv(PSF, clear_img_s, h_num=h_num, w_num=w_num)

        if save_psf_all:
            save_filename = "iter" + str(self.iteration) + '.png'
            utils.save_image(PSF_draw.reshape(h_num*w_num, 3, PSF_draw.size(2), PSF_draw.size(3)), os.path.join(self.path.demo_root, 'all_psf_'+save_filename), nrow=w_num,
                             normalize=True, sacle_each=True)

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
                    img_pad[:, :,
                    index_h * patch_length + pad_whole - pad:(index_h + 1) * patch_length + pad_whole + pad,
                    index_w * patch_length + pad_whole - pad:(index_w + 1) * patch_length + pad_whole + pad]
        outputs = F.conv2d(inputs_pad, PSF_draw, stride=1, groups=C * h_num * w_num)

        for index_h in range(0, h_num):
            for index_w in range(0, w_num):
                blur_img[:, :, index_h * patch_length:(index_h + 1) * patch_length,
                index_w * patch_length:(index_w + 1) * patch_length] = outputs[:, (index_w + index_h * w_num) * C:(
                                                                                                                              index_w + index_h * w_num + 1) * C]
        # for i in range(N):
        #     blur_img[i] = utils_ga.make_grid(outputs[i].reshape(h_num * w_num, C, patch_length, patch_length), padding=0, nrow=w_num)
        time3 = time.time()
        print('patch: {time}'.format(time=patch_length))
        print('fast_conv_time: {time}'.format(time=time3 - time1))

        return blur_img, PSF_draw

    def super_fast_patch_wise_conv(self, psf, img, h_num=100, w_num=100):
        patch_length = self.patch_length
        view_pos = self.view_pos
        time1 = time.time()
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
        psf_used1 = self.my_rotate(psf_used, interpolation=InterpolationMode.BILINEAR, angle=rotate_theta)
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
        # print('super_fast_conv_time: {time}'.format(time=time3 - time1))

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
        rescaled_theta = theta.transpose(1, 2) / torch.tensor([0.5 * w, 0.5 * h], dtype=theta.dtype,
                                                              device=theta.device)
        output_grid = base_grid.view(n, oh * ow, 3).bmm(rescaled_theta)
        grid = output_grid.view(n, oh, ow, 2)
        # _apply_grid_transform(img, grid, interpolation, fill=fill)
        output = torch.grid_sampler(img, grid, 0, 0, False)
        return output
        # img = grid_sample(img, grid, mode=interpolation.value, padding_mode="zeros", align_corners=False)
