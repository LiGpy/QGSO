import os
import numpy as np
import torch
import copy
from torchvision import io
from torchvision import utils
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, T_co
from post_prossess.Models.metrics import calculate_psnr, calculate_ssim, calculate_lpips
import torch.nn.functional as F


class EMA:
    def __init__(self, beta=0.9999):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class BasicModel:
    def __init__(self, network, optimizer_data, train, img_path, ema_scheduler=None,
                 optimizer_scheduler=None, device=torch.device('cuda:0'), model_path=None, dtype=torch.float32):
        self.metrics_all = None
        self.calc_lpips = None
        self.train = train
        self.img_path = img_path
        self.naf_iter = 0
        self.netG = network
        if model_path is not None:
            self.netG.load_state_dict(torch.load(model_path, map_location=device))
        self.device = device
        self.dtype = dtype
        if ema_scheduler is not None:
            self.ema_scheduler = ema_scheduler
            self.netG_EMA = copy.deepcopy(self.netG)
            self.EMA = EMA(beta=self.ema_scheduler['ema_decay'])
        else:
            self.ema_scheduler = None
        self.img_path = img_path
        self.blur_root = img_path["blur_root"]
        self.clear_root = img_path["clear_root"]
        self.blur_val_root = img_path["blur_val_root"]
        self.clear_val_root = img_path["clear_val_root"]
        self.save_root = img_path["save_root"]
        self.demo_root = img_path["demo_root"]
        self.blur_val_root_no_gt = img_path["blur_val_root_no_gt"]

        if self.blur_val_root_no_gt is not None:
            img_s_val_blur_no_gt = sorted(os.listdir(self.blur_val_root_no_gt))
            self.list_val_blur_no_gt = []
            for i in img_s_val_blur_no_gt:
                self.list_val_blur_no_gt.append(os.path.join(self.blur_val_root_no_gt, i))

        img_s_clear = sorted(os.listdir(self.clear_root))
        self.list_clear = []
        for i in img_s_clear:
            self.list_clear.append(os.path.join(self.clear_root, i))

        img_s_blur = sorted(os.listdir(self.blur_root))
        self.list_blur = []
        for i in img_s_blur:
            self.list_blur.append(os.path.join(self.blur_root, i))

        img_s_val_blur = sorted(os.listdir(self.blur_val_root))
        self.list_val_blur = []
        for i in img_s_val_blur:
            self.list_val_blur.append(os.path.join(self.blur_val_root, i))

        img_s_val_clear = sorted(os.listdir(self.clear_val_root))
        self.list_val_clear = []
        for i in img_s_val_clear:
            self.list_val_clear.append(os.path.join(self.clear_val_root, i))

        self.optG = torch.optim.Adam(list(filter(lambda p: p.requires_grad, self.netG.parameters())), **optimizer_data)
        # self.opt_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optG, **optimizer_scheduler)
        self.phase = 'train'
        self.all_iter = self.train["n_iter"]
        self.test_iter = self.train["test_iter"]
        self.val_iter = self.train["val_iter"]
        self.now_iter = 0
        self.writer = None
        self.big_clear_img_s = None
        self.big_blur_img_s = None

    def get_current_visuals(self, phase='train'):
        pass

    def save_current_results(self):
        pass

    def train_step(self):
        self.now_iter = 0
        self.metrics_all = []
        for iteration in range(self.all_iter):

            self.now_iter = iteration
            gt_image, cond_image = self.set_input_fast(phase='train')
            self.optG.zero_grad()
            output_img = self.netG(cond_image)
            loss_net = F.mse_loss(output_img, gt_image)
            loss_net.backward()
            self.optG.step()
            if iteration % 10 == 0:
                self.writer.add_scalar("loss", loss_net.detach(), iteration)
            print('iter:{iteration}_loss: {loss}'.format(loss=loss_net.item(), iteration=iteration))
            if self.ema_scheduler is not None:
                if iteration > self.ema_scheduler['ema_start'] and iteration % self.ema_scheduler['ema_iter'] == 0:
                    self.EMA.update_model_average(self.netG_EMA, self.netG)
            if (iteration + 1) % self.test_iter == 0:
                print('test...')
                self.test(test_num=1)
            if (iteration + 1) % self.val_iter == 0:
                print('val...')
                self.val_step()
                if (iteration + 1) / self.val_iter >= 2:
                    if self.metrics_all[-1] - self.metrics_all[-2] >= -0.001:
                        break
        print('final_val...')
        self.val_step_no_gt()

    def set_input_fast(self, phase='train'):
        clear_img_s = None
        blur_img_s = None
        edof_num = round(len(self.list_blur) / len(self.list_clear))
        self.big_clear_img_s = []
        self.big_blur_img_s = []
        if phase == 'train':
            pre_b_size = self.train["pre_b_size"]
            for i in range(pre_b_size):
                index = np.random.randint(0, len(self.list_clear))
                img_path_clear = self.list_clear[index]
                img_path_blur = self.list_blur[index * edof_num: (index + 1) * edof_num]
                for j in range(edof_num):
                    self.big_clear_img_s.append(io.read_image(img_path_clear).float() / 255)
                    self.big_blur_img_s.append(io.read_image(img_path_blur[j]).float() / 255)
            b_size = self.train["b_size"]
            img_size = self.train["img_size"]
            # index = np.random.randint(0, len(self.list_clear))
            blur_img_s = torch.zeros([b_size * edof_num, 3, img_size, img_size]).to(self.device)
            clear_img_s = torch.zeros([b_size * edof_num, 3, img_size, img_size]).to(self.device)
            for i in range(b_size):
                index = np.random.randint(0, pre_b_size)
                image_clear = self.big_clear_img_s[index * edof_num: (index + 1) * edof_num]
                image_blur = self.big_blur_img_s[index * edof_num: (index + 1) * edof_num]
                c, h, w = image_clear[0].shape
                h_randint = np.random.randint(0, h - img_size)
                w_randint = np.random.randint(0, w - img_size)
                for j in range(edof_num):
                    clear_img_s[i * edof_num + j] = image_clear[j][:, h_randint:h_randint + img_size,
                                                    w_randint:w_randint + img_size]
                    blur_img_s[i * edof_num + j] = image_blur[j][:, h_randint:h_randint + img_size,
                                                   w_randint:w_randint + img_size]
        if phase == 'test':
            index = np.random.randint(0, len(self.list_clear))
            img_path_clear = self.list_clear[index]
            img_path_blur = self.list_blur[index * edof_num: (index + 1) * edof_num]
            for j in range(edof_num):
                self.big_clear_img_s.append(io.read_image(img_path_clear).float() / 255)
                self.big_blur_img_s.append(io.read_image(img_path_blur[j]).float() / 255)
            c, h, w = self.big_clear_img_s[0].shape
            blur_img_s = torch.zeros([edof_num, c, h, w]).to(self.device)
            clear_img_s = torch.zeros([edof_num, c, h, w]).to(self.device)
            for j in range(edof_num):
                blur_img_s[j] = self.big_blur_img_s[j]
                clear_img_s[j] = self.big_clear_img_s[j]

        return clear_img_s, blur_img_s

    def val_step(self, save=False, is_patch=False):
        calc_lpips = calculate_lpips(device=self.device)
        edof_num = round(len(self.list_val_blur) / len(self.list_val_clear))
        with torch.no_grad():
            PSNR = []
            SSIM = []
            LPIPS = []
            NIQE = []
            FID = []
            for index in range(len(self.list_val_clear)):
            # for index in range(2):
                img_path_clear = self.list_val_clear[index]
                img_path_blur = self.list_val_blur[index * edof_num: (index + 1) * edof_num]
                clear_img_s = (io.read_image(img_path_clear).float() / 255).unsqueeze(0).to(self.device)
                save_path = self.save_root
                for j in range(edof_num):
                    blur_img_s = (io.read_image(img_path_blur[j]).float() / 255).unsqueeze(0).to(self.device)
                    output = self.netG_EMA(blur_img_s)
                    if save:
                        utils.save_image(output, save_path + str(index) + '_' + str(j) + 'deblur_val.png')
                        utils.save_image(blur_img_s, save_path + str(index) + '_' + str(j) + 'blur_val.png')
                        if j == 0:
                            utils.save_image(clear_img_s, save_path + str(index) + 'gt_val.png')
                    psnr_single = calculate_psnr(output, clear_img_s)
                    ssim_single = calculate_ssim(output, clear_img_s)
                    LPIPS_single = calc_lpips.calc(output / 0.5 - 1, clear_img_s / 0.5 - 1)
                    print('index:{i}_{j}'.format(i=index, j=j))
                    print('PSNR:{psnr}'.format(psnr=psnr_single))
                    print('SSIM:{ssim}'.format(ssim=ssim_single))
                    print('LPIPS:{lpips}'.format(lpips=LPIPS_single))
                    PSNR.append(psnr_single)
                    SSIM.append(ssim_single)
                    LPIPS.append(LPIPS_single)

            psnr = sum(PSNR) / len(PSNR)
            ssim = sum(SSIM) / len(SSIM)
            lpips = sum(LPIPS) / len(LPIPS)
            self.metrics_all.append((40 - psnr) / 20 + (1 - ssim) + lpips)
            self.writer.add_scalar("psnr", psnr, self.now_iter)
            self.writer.add_scalar("ssim", ssim, self.now_iter)
            self.writer.add_scalar("lpips", lpips, self.now_iter)
            print('mean_PSNR:{psnr}'.format(psnr=psnr))
            print('mean_SSIM:{ssim}'.format(ssim=ssim))
            print('mean_LPIPS:{lpips}'.format(lpips=lpips))

    def val_step_no_gt(self, save=True, is_patch=False):
        with torch.no_grad():
            for index in range(len(self.list_val_blur_no_gt)):
                save_path = self.save_root
                blur_img_s = (io.read_image(self.list_val_blur_no_gt[index]).float() / 255).unsqueeze(0).to(self.device)
                output = self.netG_EMA(blur_img_s)
                if save:
                    utils.save_image(output, save_path + str(index) + 'no_gt_deblur_val.png')
                    utils.save_image(blur_img_s, save_path + str(index) + 'no_gt_blur_val.png')

    def test(self, test_num=1):
        edof_num = round(len(self.list_blur) / len(self.list_clear))
        with torch.no_grad():
            for i in range(test_num):
                gt_image, cond_image = self.set_input_fast(phase='test')
                n = gt_image.size(0)
                for j in range(n):
                    name = 'iter{num}_'.format(num=self.now_iter)
                    save_path = os.path.join(self.save_root + name)
                    if j % edof_num == 0:
                        utils.save_image(gt_image[j].unsqueeze(0), save_path + str(i) + 'gt.png')
                    x_init = self.netG_EMA(cond_image[j].unsqueeze(0))
                    utils.save_image(x_init, save_path + str(i) + '_' + str(j) + 'deblur.png')
                    utils.save_image(cond_image[j].unsqueeze(0), save_path + str(i) + '_' + str(j) + 'blur.png')

            self.save_everything(model_root=self.save_root)

    def load_networks(self):
        """ save pretrained model and training state, which only do on GPU 0. """
        pass

    def save_everything(self, model_root='./model_path/20230304'):
        """ load pretrained model and training state. """
        netG_label = 'swinir'
        # torch.save(self.netG.state_dict(), model_root + netG_label + str(self.now_iter) + '.pt')
        if self.ema_scheduler is not None:
            net_ema_label = netG_label + '_ema'
            torch.save(self.netG_EMA.state_dict(), model_root + net_ema_label + '.pt')

        # state = {'iter': self.now_iter}
        # save_filename = '{}.state'.format(self.epoch)
        # save_path = os.path.join(self.opt['path']['checkpoint'], save_filename)
        # torch.save(state, save_path)
