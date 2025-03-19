import argparse
import time
import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast, GradScaler
import diffoptics as do
from diffoptics.simulation_func import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from post_prossess.Models.metrics import calculate_psnr, calculate_ssim, calculate_lpips
from post_prossess.Models.loss_basic import mse_loss, l1_loss, LpipsLoss, WeightLoss
# 2345789
# 246789

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_op', type=int, default=0, help='0 1 2 3 4 5')
    parser.add_argument('--num_3p', type=int, default=1, help='1 2 3')
    parser.add_argument('--source', type=str, default='gso', help='codev gso')
    parser.add_argument('--num_dev', type=int, default=0, help='0 1 2')
    parser.add_argument('--model_name', type=str, default='SwinIR', help='SwinIR NAFNet')
    parser.add_argument('--mode', type=str, default='e2e', help='e2e img')
    parser.add_argument('--run_place', type=str, default='cloud', help='leisun cloud')
    args = parser.parse_args()
    lens = do.LensGroup()
    torch.set_num_threads(8)
    """
    初始化镜头，超参数配置
    """
    num_op = args.num_op  # 0 1 2 3 4
    num_3p = args.num_3p  # 0 1 2
    source = args.source  # 'codev'、'gso'
    num_dev = args.num_dev  # 0 1 2
    # SwinIR NAFNet
    model_name = args.model_name  # 'SwinIR'
    mode = args.mode  # 'spot'、'img' 和 'e2e'
    run_place = args.run_place  # 'leisun'、'cloud'

    txt_name = source + str(num_3p)
    device = torch.device('cuda:'+ str(num_dev))
    if mode == 'spot':
        label = str(num_op) + '_' + mode
    else:
        label = str(num_op) + '_' + model_name + '_' + mode

    base_root = './EDOF_data/{txt_name}/{num}/{x}/'.format(num=num_op, x=label, txt_name=txt_name)
    # 云服务器
    if run_place == 'leisun':
        lens.path.train_clear_root = '/cluster/work/cvl/leisun/gaoyao/EDOF_data/train_clear/'
        lens.path.val_clear_root = '/cluster/work/cvl/leisun/gaoyao/EDOF_data/val_clear/'
        lens.path.optics_read_path = './{txt_name}.txt'.format(txt_name=txt_name)
        base_root = '/cluster/work/cvl/leisun/gaoyao/EDOF_data/{txt_name}/{num}/{x}/'.format(num=num_op, x=label, txt_name=txt_name)
    elif run_place == 'cloud':
        lens.path.train_clear_root = './EDOF_data/train_clear/'
        lens.path.val_clear_root = './EDOF_data/val_clear/'
        lens.path.optics_read_path = './init_structure/{txt_name}.txt'.format(txt_name=txt_name)
        base_root = './EDOF_data/{txt_name}/{num}/{x}/'.format(num=num_op, x=label, txt_name=txt_name)

    if num_3p == 1:
        lens.basics.z_object = -10000  # 物面位置
        lens.basics.z_object_edof = [-100000, -10000, -5000]
        lens.basics.min_F_num = 2.5  # 最小F数，仿真不用
        lens.simulation.fov_hand_set = torch.Tensor([20])[0].to(device)  # 定义视场角
        lens.constraint.distort = (-0.02, 0.02)
        lens.constraint.img = (-0.04, 0.04)
        lens.constraint.effl = (38, 42)
    elif num_3p == 2:
        lens.basics.z_object = -5000  # 物面位置
        lens.basics.z_object_edof = [-50000, -5000, -1500]
        lens.basics.min_F_num = 3.5  # 最小F数，仿真不用
        lens.simulation.fov_hand_set = torch.Tensor([24])[0].to(device)  # 定义视场角
        lens.constraint.distort = (-0.05, 0.05)
        lens.constraint.img = (-0.04, 0.04)
        lens.constraint.effl = (30, 34)
    elif num_3p == 3:
        lens.basics.z_object = -1000  # 物面位置
        lens.basics.z_object_edof = [-5000, -1000, -500]
        lens.basics.min_F_num = 4.0  # 最小F数，仿真不用
        lens.simulation.fov_hand_set = torch.Tensor([32])[0].to(device)  # 定义视场角
        lens.constraint.distort = (-0.08, 0.08)
        lens.constraint.img = (-0.04, 0.04)
        lens.constraint.effl = (21, 25)

    # 仿真参数配置
    lens.diff.pre_epochs = 200
    lens.diff.epochs = 200
    lens.diff.look_per_epochs = 5
    lens.simulation.wave_path = './wave.mat'  # lens.simulation.wave_path = None
    lens.simulation.wave_sample_num = 5  # 必须为奇数
    lens.simulation.device = device
    lens.simulation.patch_length = 64
    lens.simulation.b_size = 1
    lens.simulation.fov_num = 3
    lens.simulation.enp_num = 64
    lens.simulation.psf_num = 31
    lens.simulation.psf_num_max = 41
    lens.simulation.img_size = (1280, 1920)
    lens.simulation.img_pixel_size = 0.012394
    lens.simulation.flag_auto_calc = True
    lens.simulation.flag_simu_relative_illumination = False
    lens.simulation.flag_simu_distortion = False  # 是否仿真畸变，可以仿真棋盘格时用来看效果，仿真图像时可以设置为True， 端到端训练时可设置为False
    lens.simulation.disturbance = 0.  # 光学参数设置的扰动幅度，比如0.1就是扰动幅度10%
    lens.simulation.use_isp = True

    # 波长定义，单位[nm]，三个通道定义各自波长个数和对应权重，近似传感器光谱响应曲线，此处每个通道只有一个波长权重为1，但实际上每个通道可以定义任意个波长和相应权重。
    if lens.simulation.wave_path is not None:
        lens.simulation.wavelength = lens.simulation.sample_wave()
    else:
        lens.simulation.wavelength = {
            "R": ([656.3], [1]),
            "G": ([587.6], [1]),
            "B": ([486.1], [1]),
        }

    # 文件路径配置
    # 304
    # lens.path.optics_read_path = './init_structure/optic_selected_data.txt'
    # lens.path.train_clear_root = '/mnt/gy/train_clear/'
    # lens.path.val_clear_root = '/mnt/gy/val_clear/'
    # base_root = '/mnt/gy/EDOF_data/{num}/{x}/'.format(num=num_op, x=label)

    lens.path.optics_write_path = base_root + '{x}.txt'.format(x=label)
    lens.path.train_blur_root = base_root
    lens.path.val_blur_root = base_root
    lens.path.demo_root = base_root
    lens.path.save_root = base_root

    # lens.path.demo_root = './init_structure/'
    # lens.path.save_root ='./init_structure/'

    lens.diff.writer = SummaryWriter(log_dir=base_root + 'loss_dir')
    lens.diff.model_root = base_root

    # 光学系统基本参数配置
    lens.basics.MAX_R = 20  # 最大尺寸半径，仿真不用
    lens.diff.weight_loss_constraint = 1
    lens.diff.weight_loss_network = 100
    lens.diff.weight_loss_rms = 10
    lens.diff.require_grad = True
    lens.diff.calc_lpips = calculate_lpips(device=device)
    lens.diff.ao_path = './test_plate_ao.mat'
    lens.diff.tu_path = './test_plate_tu.mat'

    lens.diff.train_rate = 1000
    lens.diff.test_iter = 2000
    lens.diff.val_per_epochs = 1
    lens.diff.post_b_size = 4
    lens.diff.post_img_size = 256
    lens.diff.model = lens.set_model(model_name)

    if mode == 'e2e':
        lens.diff.flag_use_post = True
    else:
        lens.diff.flag_use_post = False

    if mode == 'spot':
        lens.diff.only_spot = True
    else:
        lens.diff.only_spot = False

    # 物理约束边界定义
    lens.constraint.thi_glass_cen = (4.0, 15.0)
    lens.constraint.thi_air_cen = (1, 15)
    lens.constraint.c = (-0.1, 0.1)
    lens.constraint.k = None
    lens.constraint.aspheric = None
    lens.constraint.material_n = (1.51, 1.76)
    lens.constraint.material_V = (27.5, 71.3)
    lens.constraint.thi_glass_border = (5, 15)
    lens.constraint.thi_air_border = (1, 15)
    lens.constraint.bfl = (18, 60)
    lens.constraint.z_sensor = (20, 60)

    lens.constraint.aper_d = (1, 15)
    lens.constraint.edge_ray = 99999

    # 绘制小测试
    # for i in range(11):
    #     lens.basics.surfaces, lens.basics.materials, lens.basics.aper_index, lens.basics.z_sensor = lens.read_data(lens.path.optics_read_path
    #                                                                                                 , device, now_epochs=i)
    #     is_normal = lens.calc_order1(flag_diff=False)
    #     # 画光线追迹图
    #     lens.draw_all(M=31, name="raytrace_iter{num}".format(num=i))

    now_epochs = 0  # 如果要从txt文件里读入数据，这个参数决定读取哪一组数据
    if lens.path.optics_read_path is not None:
        lens.basics.surfaces, lens.basics.materials, lens.basics.aper_index, lens.basics.z_sensor = lens.read_data(lens.path.optics_read_path, device, now_epochs=num_op)
    else:
        lens.basics.surfaces = [
            do.Aspheric(7.9734, 0.0, c=1 / 20.2554, k=0.0243, device=device),
            do.Aspheric(4.9548, 7.4059, c=1 / 9.8712, device=device),
            do.Aspheric(3.7139, 12.3267, c=1 / 1e10, device=device),
            do.Aspheric(2.4768, 21.0351, c=1 / 372.5677, device=device),
            do.Aspheric(1.9536, 23.0483, c=1 / 1e10, device=device),
        ]  # 透镜面定义（包括孔径光阑在内）
        lens.basics.materials = [
            do.Material(device=device, name='air'),
            do.Material(device=device, name='my_glass2'),
            do.Material(device=device, name='air'),
            do.Material(device=device, name='air'),
            do.Material(device=device, name='my_glass3'),
            do.Material(device=device, name='air'),
        ]  # 透镜材料定义

    lens.diff.set_diff_parameters_all(lens)
    # lr_rate = 1e-3
    # lr = (lr_rate * (lens.constraint.c[1] - lens.constraint.c[0]), 5 * lr_rate * (lens.constraint.thi_glass_cen[1] - lens.constraint.thi_glass_cen[0]),
    #       lr_rate * 5 * (lens.constraint.material_n[1] - lens.constraint.material_n[0]), lr_rate * 5 * (lens.constraint.material_V[1] - lens.constraint.material_V[0]))
    lr = (0.0002, 0.02, 0.001, 0.2)
    lens.diff.set_optimizer_optics('Adam', beta=(0.5, 0.9), lr=lr, lens=lens)
    # lens.diff.set_optimizer_optics('SGD', lr=(2e-5, 1e-3, 2e-4, 4e-3), constraint=lens.constraint)
    lens.diff.post_iter = 0
    lens.diff.post_rate = 1

    """
    开始优化
    """
    while now_epochs <= lens.diff.epochs:
        tic = time.time()  # 运行时间计算
        """
        仿真准备工作，相关参数初始化计算
        """
        img_model = lens.diff.model
        lens.diff.now_epochs = lens.simulation.now_epochs = now_epochs
        lens.diff.set_period(materials=lens.basics.materials, surfaces=lens.basics.surfaces, aper_idx=lens.basics.aper_index)
        if lens.diff.stop_flag:
            lens.write_optics_data(save_path=lens.path.optics_write_path, loss=(torch.Tensor(lens.diff.loss_all).to(lens.diff.device)[lens.diff.last_epochs:]).min().item(), end_flag=True)
            break

        # lens.diff.period_all = 'post'
        # lens.diff.flag_use_post = True
        if lens.diff.period_all == 'post':
            lens.diff.require_grad = False
            # for i in range(len(lens.diff.parameters_all)):
            #     lens.diff.parameters_all[i].requires_grad = False

        # if now_epochs >= lens.diff.pre_epochs:
        #     lens.diff.flag_use_post = True

        if lens.diff.require_grad:
            lens.diff.adam_zero_grad()

        if lens.diff.flag_use_post and now_epochs == 0:
            lens.diff.post_rate = 4
        elif now_epochs >= lens.diff.pre_epochs:
            lens.diff.post_rate = 1
        else:
            lens.diff.post_rate = 1

        with torch.no_grad():
            # 图像数据初始化
            lens.simulation.set_path(lens.path.train_clear_root)
            # 自动计算一阶参数
            is_normal = lens.calc_order1(flag_diff=False)
        # 画光线追迹图
        lens.draw_all(M=11)
        if lens.diff.only_spot:
            lens.diff.loss_rms = lens.diff.weight_loss_rms * (lens.calc_rms() ** 1)
            lens.diff.loss_constraint = lens.calc_loss_constraint()
            time2 = time.time()
            print('time_rms:{num}'.format(num=time2 - tic))
            lens.diff.step_only_spot()
            lens.write_optics_data(save_path=lens.path.optics_write_path, loss=(lens.diff.loss_rms + lens.diff.loss_constraint * lens.diff.weight_loss_constraint).item())
            print('now_epochs:{num}'.format(num=now_epochs))
            print('loss_constraint: {loss}'.format(loss=lens.diff.loss_constraint.item() * lens.diff.weight_loss_constraint))
            print('loss_rms: {loss}'.format(loss=lens.diff.loss_rms))
            print('datas: {datas}'.format(datas=lens.diff.datas))
            print('grads: {grads}'.format(grads=lens.diff.grads))
            print('vignetting: {vignetting}'.format(vignetting=lens.order1.vignetting_factors))
        else:
            # 生成PSF, PSF尺寸统一，并考虑色差
            lens.calc_psf(use_spot=True, save_main_psf=False, require_grad=lens.diff.require_grad)
            lens.diff.loss_constraint = lens.calc_loss_constraint()
            # 点扩散函数尺寸越小，代表像差越小，可用更大的图像块，节省内存
            time2 = time.time()
            print('time_psf:{num}'.format(num=time2 - tic))
            loss_vis = []
            for num_per_optics in range(int(lens.diff.train_rate * lens.diff.post_rate)):
                if lens.diff.flag_use_post:
                    img_model.optG.zero_grad()
                # PSF尺寸统一
                time1 = time.time()

                lens.uni_psf_pre(save_rgb_psf=True, count=num_per_optics)
                # 卷积得到图像
                # time2 = time.time()
                clear_img_list, blur_img_list, filename_s = lens.simulation.simulation_to_img(lens.order1.PSF, lens.order1.fov_pos,
                                                                                              lens.path, now_epochs=now_epochs,
                                                                                              flag_simulation_random=True)
                # time3 = time.time()
                loss_post = lens.diff.train_end2end(clear_img_list, blur_img_list, count=num_per_optics)

                time4 = time.time()
                # print(time2 - time1)
                # print(time3 - time2)
                # print(time4 - time1)
                loss_vis.append(loss_post.item())

                if lens.diff.flag_use_post:
                    img_model.optG.step()
                print('optics:{epoch}_post:{count}_loss: {loss}'.format(loss=loss_post.item(), epoch=now_epochs, count=num_per_optics))

            if lens.diff.require_grad:
                lens.diff.step(lens.order1.raw_psf, lens.order1.d_PSF)

            print('now_epochs:{num}'.format(num=now_epochs))
            print('loss_constraint: {loss}'.format(loss=lens.diff.loss_constraint.item() * lens.diff.weight_loss_constraint))
            print('loss_post: {loss}'.format(loss=sum(loss_vis) / len(loss_vis)))
            print('datas: {datas}'.format(datas=lens.diff.datas))
            print('grads: {grads}'.format(grads=lens.diff.grads))
            print('vignetting: {vignetting}'.format(vignetting=lens.order1.vignetting_factors))
            toc = time.time()
            print('time: {time}'.format(time=toc - tic))

            # 验证环节
            print('val...')
            lens.simulation.set_path(lens.path.val_clear_root)
            PSNR = []
            SSIM = []
            LPIPS = []
            LOSS_FINAL = []
            for index in range(len(lens.simulation.img_s)):
                clear_img_list, blur_img_list, filename_s = lens.simulation.simulation_to_img(lens.order1.PSF,
                                                                                              lens.order1.fov_pos,
                                                                                              lens.path, now_epochs=now_epochs,
                                                                                              flag_simulation_random=False,
                                                                                              specific_index=[index])
                psnr_single, ssim_single, LPIPS_single, loss_final_single = lens.diff.val_end2end(clear_img_list, blur_img_list, index=index, save=False)
                print(index)
                PSNR.append(psnr_single)
                SSIM.append(ssim_single)
                LPIPS.append(LPIPS_single)
                LOSS_FINAL.append(loss_final_single)
            psnr = sum(PSNR) / len(PSNR)
            ssim = sum(SSIM) / len(SSIM)
            lpips = sum(LPIPS) / len(LPIPS)
            loss_final = sum(LOSS_FINAL) / len(LOSS_FINAL)

            lens.diff.writer.add_scalar("psnr", psnr, lens.diff.post_iter)
            lens.diff.writer.add_scalar("ssim", ssim, lens.diff.post_iter)
            lens.diff.writer.add_scalar("lpips", lpips, lens.diff.post_iter)
            lens.diff.writer.add_scalar("loss_e2e", loss_final, lens.diff.post_iter)
            print('mean_PSNR:{psnr}'.format(psnr=psnr))
            print('mean_SSIM:{ssim}'.format(ssim=ssim))
            print('mean_LPIPS:{lpips}'.format(lpips=lpips))
            print('loss_final:{l}'.format(l=loss_final))

            img_model.save_everything(model_root=lens.diff.model_root)
            lens.simulation.set_path(lens.path.train_clear_root)

            lens.diff.loss_all.append(loss_final)

            if lens.diff.period_all != 'post':
                lens.write_optics_data(save_path=lens.path.optics_write_path, loss=loss_final)
        now_epochs += 1


if __name__ == '__main__':
    main()