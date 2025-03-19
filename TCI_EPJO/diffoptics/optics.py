import torch
import time
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import meshio
import torch.nn.functional as F
from torchvision.transforms.functional import InterpolationMode, resize
from torchvision import utils
from .order1_func import Order1
from .constraint_func import Constraint
from .simulation_func import Simulation
from .diff_func import Diff
from .basics import *
from torchvision import transforms
from post_prossess.Models.model_init import def_model


# 绘制透镜界面：316~568行
# 光线采样：610~875行
class Path:
    def __init__(self):
        self.optics_read_path = None
        self.train_clear_root = None
        self.train_blur_root = None
        self.val_clear_root = None
        self.val_blur_root = None
        self.demo_root = None
        self.save_root = None
        self.model_path = None


class Basics:
    def __init__(self):
        self.z_object = None
        self.z_object_edof = None
        self.z_sensor = None
        self.surfaces = None
        self.materials = None
        self.aper_index = None
        self.min_F_num = None
        self.MAX_R = None


class LensGroup:
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

    def set_model(self, net_name):
        self.diff.net_name = net_name
        model = def_model(net_name, device=self.simulation.device, clear_root=self.path.train_clear_root,
                          blur_root=self.path.train_blur_root, clear_val_root=self.path.val_clear_root,
                          blur_val_root=self.path.val_blur_root, save_root=self.path.save_root,
                          demo_root=self.path.demo_root, model_path=self.path.model_path)

        return model

    def calc_order1(self, calc_zoom_rate=False, flag_diff=False, flag_calc_exit=False, resume=True):
        t1 = time.time()
        wavelength_all = self.simulation.wavelength
        self.order1.wave_data = {
            "all": wavelength_all['R'][0] + wavelength_all['G'][0] + wavelength_all['B'][0],
            "all_weight": wavelength_all['R'][1] + wavelength_all['G'][1] + wavelength_all['B'][1],
            "main_RGB": [np.median(wavelength_all['R'][0]), np.median(wavelength_all['G'][0]),
                         np.median(wavelength_all['B'][0])],
            "main": float(np.median(wavelength_all['G'][0]))
        }
        self.order1.img_r = (self.simulation.img_size[0] ** 2 + self.simulation.img_size[
            1] ** 2) ** 0.5 * self.simulation.img_pixel_size / 2

        self.order1.aperture_max = self.calc_max_aper(
            wavelength=self.order1.wave_data["main"], resume=resume)
        if self.simulation.flag_auto_calc:
            self.basics.surfaces[self.basics.aper_index].r = self.order1.aperture_max

        if self.simulation.fov_hand_set is not None:
            self.order1.fov_max = self.simulation.fov_hand_set
        else:
            self.order1.fov_max = self.calc_max_fov(wavelength=self.order1.wave_data["main"])
        if calc_zoom_rate:
            self.order1.zoom_rate = self.calc_room_rate(wavelength=self.order1.wave_data["main"], view=0.0)  # 计算近轴放大率
        if flag_diff:
            self.diff.init_raytrace(device=self.simulation.device)
        if self.simulation.flag_auto_calc:
            resize_normal = self.resize_lens(wavelength=self.order1.wave_data["main"], view=self.order1.fov_max.item(),
                                             vignetting=1)
        else:
            resize_normal = True
        self.order1.fov_samples, self.order1.fov_pos = self.calc_fov_samples(self.order1.fov_max,
                                                                             self.order1.wave_data["main_RGB"],
                                                                             non_uniform=False)
        if flag_calc_exit:
            self.calc_exit_pupil()

        # 3、计算得到每个视场下主波长光线在第一面的入瞳坐标与相对照度（渐晕）因子
        self.order1.enp_xy, self.order1.vignetting_factors = self.calc_enp_all_fov(
            wavelength=self.order1.wave_data["main"])

        is_normal = resize_normal
        t2 = time.time()  # 运行时间计算
        print('time_order1:{time}'.format(time=t2 - t1))
        return is_normal

    def disturb_func(self, disturbance):
        device = self.simulation.device
        surfaces = self.basics.surfaces
        materials = self.basics.materials
        for i in range(len(surfaces)):
            surface = surfaces[i]
            surface.c = surface.c * (1 + (torch.rand(1) - 0.5) / 0.5 * disturbance).to(device)
            surface.d = surface.d * (1 + (torch.rand(1).item() - 0.5) / 0.5 * disturbance)
        for i in range(len(materials)):
            mater = materials[i]
            if mater.n > 1.1:
                mater.n = mater.n * (1 + (torch.rand(1) - 0.5) / 0.5 * disturbance).to(device)
                mater.V = mater.V * (1 + (torch.rand(1) - 0.5) / 0.5 * disturbance).to(device)

    # ------------------------------------------------------------------------------------
    # 画图
    # ------------------------------------------------------------------------------------
    def draw_all(self, M=11, name=None):
        plt.ioff()
        num = self.diff.now_epochs
        with torch.no_grad():
            # 光线追迹可视化
            ax, fig = self.plot_setup2D_with_trace_whole(self.order1.fov_samples, self.order1.wave_data["main"], M=M)
            ax.axis('off')
            if name is None:
                item = "raytrace_iter{num}".format(num=num)
            else:
                item = name
            # ax.set_title(item)
            name = self.path.demo_root + item
            fig.savefig(name, bbox_inches='tight')
            plt.close()

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

    def plot_setup2D(self, ax=None, fig=None, show=True, color='k', with_sensor=True):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        materials = self.basics.materials
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
            draw_aperture(ax, aperture_distance, aperture_radius, color)
        else:
            # draw sensor plane
            if with_sensor:
                try:
                    tmpr, _ = self.order1.img_r, self.basics.z_sensor
                except AttributeError:
                    with_sensor = False

            if with_sensor:
                surfaces.append(Aspheric(self.order1.img_r, self.basics.z_sensor, 0.0))

            draw_aperture(ax, aperture_distance, aperture_radius, color)
            # 绘制透镜表面
            for i, s in enumerate(surfaces):
                if (i + 1) < len(surfaces) and materials[i].A < 1.0003 and materials[
                    i + 1].A < 1.0003 and i == self.basics.aper_index:
                    continue
                r = torch.linspace(-s.r, s.r, s.APERTURE_SAMPLING, device=device)  # aperture sampling
                z = s.surface_with_offset(r, torch.zeros(len(r), device=device))
                plot(ax, z, r, color)

            # 绘制边界
            s_prev = []
            for i, s in enumerate(surfaces):
                if materials[i].A < 1.0003:  # 空气
                    s_prev = s
                else:
                    r_prev = s_prev.r
                    r = s.r
                    sag_prev = s_prev.surface_with_offset(r_prev, 0.0).squeeze(0)
                    sag = s.surface_with_offset(r, 0.0).squeeze(0)
                    z = torch.stack((sag_prev, sag))
                    x = torch.Tensor(np.array([max(r_prev, r), max(r_prev, r)])).to(device)
                    plot(ax, z, x, color)
                    plot(ax, z, -x, color)
                    if r > r_prev:
                        z = torch.stack((sag_prev, sag_prev))
                        x = torch.Tensor(np.array([r_prev, r])).to(device)
                        plot(ax, z, x, color)
                        plot(ax, z, -x, color)
                    else:
                        z = torch.stack((sag, sag))
                        x = torch.Tensor(np.array([r, r_prev])).to(device)
                        plot(ax, z, x, color)
                        plot(ax, z, -x, color)
                    s_prev = s

            # 移除传感器平面
            if with_sensor:
                surfaces.pop()

        plt.gca().set_aspect('equal', adjustable='box')
        # plt.xlabel('z [mm]')
        # plt.ylabel('r [mm]')
        # plt.title("Layout 2D")
        if show:
            plt.show()
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

    def plot_setup2D_with_trace_whole(self, views, wavelength, M=11):
        surfaces = self.basics.surfaces
        colors_list = ['m', 'b', 'c', 'g', 'y', 'r', 'k', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        ax, fig = self.plot_setup2D(show=False)
        for i, view in enumerate(views):
            ray = self.sample_ray_2D_common(wavelength, view=view, M=M, R=self.order1.enp_xy[i][2],
                                            cen=self.order1.enp_xy[i][0:2])

            oss, p = self.trace_to_sensor(ray, record=True)
            oss = torch.cat((oss, p.unsqueeze(1)), dim=1)
            if surfaces[0].surface(surfaces[0].r, 0.0) < 0:
                oss = oss[:, 1:M, :]
            ax, fig = self.plot_ray_traces(oss[[0, oss.size(0)//2, -1]], ax=ax, fig=fig, color=colors_list[i % len(colors_list)])
        return ax, fig

    # TODO: modify the tracing part to include oss
    def plot_ray_traces(self, oss, ax=None, fig=None, color='b', show=True, p=None, valid_p=None, line_width=5.0):
        """
        Plot all ray traces (oss).
        """
        device = self.simulation.device
        if ax is None and fig is None:
            ax, fig = self.plot_setup2D(show=False)
        else:
            show = False
        for i, os in enumerate(oss):
            x = os[..., 0]
            z = os[..., 2]

            # to world coordinate
            o = torch.stack((x, torch.zeros_like(x, device=device), z), dim=1).cpu().detach().numpy()
            z = o[..., 2].flatten()
            x = o[..., 0].flatten()

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

    # ------------------------------------------------------------------------------------
    # 应用程序
    # ------------------------------------------------------------------------------------
    def calc_entrance_pupil(self, wavelength=588.0, view=0.0, M=256, cen=None, R=None):
        device = self.simulation.device
        if cen is None:
            cen = [0, 0]
        ray = self.sample_ray_common(wavelength, view=view, cen=cen, R=R,
                                     M=M, sampling='grid')
        valid, ray_final, oss = self._trace(ray, record=True)
        ray_d = ray_final.d[valid]
        oss = oss[valid]
        first_px = oss[:, 0, 0]
        first_py = oss[:, 0, 1]
        x1 = first_px.min()
        x2 = first_px.max()
        y1 = first_py.min()
        y2 = first_py.max()
        ray_d_z = ray_d[:, 2]
        ray_d_xyz = (ray_d[:, 0].square() + ray_d[:, 1].square() + ray_d_z.square()).sqrt()
        cos_img_angle = (ray_d_z / ray_d_xyz).mean()
        return torch.Tensor([(x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) / 2]).to(device), len(
            first_px) * cos_img_angle ** 4

    def calc_enp_all_fov(self, wavelength=588.0):
        views = self.order1.fov_samples
        surfaces = self.basics.surfaces
        device = self.simulation.device
        R = surfaces[0].r  # [mm]
        sag = surfaces[0].surface(R, 0.0)
        vignetting_factors = torch.zeros(len(views)).to(device)
        enp_xy = torch.zeros(len(views), 3).to(device)
        R = abs((np.tan(math.radians(views.max())) * sag).item()) + R
        for i, fov in enumerate(views):
            enp_xy[i], vignetting_factors[i] = self.calc_entrance_pupil(
                wavelength=wavelength, view=fov, R=R)
        # 渐晕因子归一化(包含了相对照度的影响）
        vignetting_factors = vignetting_factors / vignetting_factors.max()
        return enp_xy, vignetting_factors

    def calc_RMS(self, wavelength=588.0, view=0.0, M=128, ray=None, record_grad=False, calc_cen=False):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        aperture_index = self.basics.aper_index
        R = surfaces[0].r  # [mm]
        sag = surfaces[0].surface(R, 0.0)
        if ray is None:
            ray = self.sample_ray_common(wavelength, view=view, cen=[(np.tan(math.radians(view)) * sag).item(), 0], R=R,
                                         M=M, sampling='grid')
        p_img = self.trace_to_sensor(ray)
        # 通过RMS直径的1.5倍(经验值)作为psf范围
        RMS, p_cen = self.rms(p_img)
        if record_grad:
            if calc_cen:
                return RMS, p_cen
            else:
                return RMS
        else:
            if calc_cen:
                return RMS.cpu().detach().numpy(), p_cen
            else:
                return RMS.cpu().detach().numpy()

    @staticmethod
    def rms(ps, units=1, ceo=False, exp=True):
        ps = ps[..., :2] * units
        ps_mean = torch.mean(ps, dim=0)
        ps_median = torch.median(ps, dim=0).values
        ps = ps - ps_mean[None, ...]  # we now use normalized
        if ceo:
            result = max(ps[:, 0].max() - ps[:, 0].min(), ps[:, 1].max() - ps[:, 1].min())
        else:
            result = torch.sqrt(torch.mean(torch.sum(ps ** 2, dim=-1)))

        # result = (1 - torch.exp(-torch.sum(ps ** 2, dim=-1) / ((0.012394*5) ** 2)).sum(dim=0) / len(ps))**2
        return result, ps_median

    def calc_effl(self, wavelength=588.0):
        device = self.simulation.device
        small_item = 1e-6
        # 主光线与边缘光线在第一面的坐标
        o = torch.Tensor([small_item, 0, 0]).to(device).unsqueeze(0).double()
        d = torch.Tensor([0, 0, 1]).to(device).unsqueeze(0).double()
        # 生成相应光线
        ray = Ray(o, d / d.square().sum().sqrt(), wavelength, device=device)
        # 输出光线，第一根为主光线，第二根为边缘光线
        valid, ray_out = self._trace(ray, stop_ind=None, record=False)
        if not valid[0]:
            raise Exception('effl ray trace is wrong!')
        o_out = ray_out.o[0]
        d_out = ray_out.d[0]
        d_out = d_out / d_out.sum()
        t = (0 - o_out[0]) / d_out[0]
        z_sensor = o_out[2] + t * d_out[2]
        t = (small_item - o_out[0]) / d_out[0]
        z_main = o_out[2] + t * d_out[2]
        if self.basics.z_sensor is None:
            self.basics.z_sensor = z_sensor  # 更正位置
        effl = z_sensor - z_main
        return effl.float()

    def calc_max_aper(self, wavelength=588.0, resume=True):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        aperture_index = self.basics.aper_index
        if resume:
            for i in range(len(surfaces) - 1):
                surfaces[i].r = min(
                    (1 / ((1 + self.basics.surfaces[i].k) * (self.basics.surfaces[i].c ** 2)) ** 0.5).item(),
                    self.basics.MAX_R)
        self.order1.effl = self.calc_effl(wavelength=wavelength).item()
        self.order1.enpd = self.order1.effl / self.basics.min_F_num
        # 孔阑中心反向追迹，得到入瞳位置
        if aperture_index == 0:
            enpp = surfaces[aperture_index].d
        else:
            o = torch.Tensor([1e-10, 0, surfaces[aperture_index - 1].d]).unsqueeze(0).to(device).double()
            d = torch.Tensor([1e-10, 0, surfaces[aperture_index - 1].d - surfaces[aperture_index].d]).unsqueeze(0).to(
                device).double()
            ray = Ray(o, d / d.square().sum().sqrt(), wavelength, device=device)
            valid, ray_out = self._trace(ray, stop_ind=aperture_index, record=False)
            if not valid[0]:
                raise Exception('ray trace is wrong!')
            o_out = ray_out.o[0]
            d_out = ray_out.d[0]
            enpp = (o_out[2] + (0 - o_out[0]) / d_out[0] * d_out[2]).item()
        self.order1.enpp = enpp
        # 正向追迹得到孔阑半径
        first_x = ((self.basics.z_object - enpp) / self.basics.z_object) * (self.order1.enpd / 2)
        o = torch.Tensor([first_x, 0, 0]).unsqueeze(0).to(device).double()
        d = torch.Tensor([first_x, 0, -self.basics.z_object]).unsqueeze(0).to(device).double()
        ray = Ray(o, d / d.square().sum().sqrt(), wavelength, device=device)
        valid, ray_out = self._trace(ray, stop_ind=aperture_index, record=False)
        return abs(ray_out.o[0, 0].item())

    def calc_max_fov(self, wavelength=588.0, M=10001, resume=True):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        aperture_index = self.basics.aper_index
        img_r = self.order1.img_r
        if resume:
            for i in range(len(surfaces) - 1):
                if i != aperture_index:
                    surfaces[i].r = min((1 / ((1 + surfaces[i].k) * (surfaces[i].c ** 2)) ** 0.5).item(),
                                        self.basics.MAX_R)
        sag = surfaces[-1].surface(surfaces[-1].r, 0.0)
        r = surfaces[-1].r
        d = surfaces[-1].d
        d_target = d + sag
        R1 = min((-(r + (r - img_r) * sag / (self.basics.z_sensor - d_target)) + 0.01), -r / 2)
        R2 = max((r + (r + img_r) * sag / (self.basics.z_sensor - d_target)) - 0.01, r / 2)
        x = torch.linspace(R1.item(), R2.item(), M, device=device)
        ones = torch.ones_like(x).to(device)
        zeros = torch.zeros_like(x).to(device)
        o = torch.stack((x, zeros, surfaces[-1].d * ones))
        origin = torch.stack((-img_r * ones, zeros, self.basics.z_sensor * ones))
        d = o - origin
        d = torch.div(d, d.norm(2, 0))
        ray = Ray(o.T, d.T, wavelength, device=device)
        valid, ray_final = self._trace(ray, record=False)
        if not valid.any():
            return None
        t_img = (self.basics.z_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
        p_img = ray_final(t_img)[valid]
        fov_max = torch.arctan(p_img[:, 0].mean() / (self.order1.enpp - self.basics.z_sensor))
        return fov_max.abs() / torch.pi * 180

    def calc_fov_samples(self, fov_max, wave_main_RGB, non_uniform=True):
        with torch.no_grad():
            if non_uniform:
                views_pre = np.degrees(
                    np.arctan(math.tan(math.radians(fov_max)) * np.linspace(0, 1, self.simulation.fov_num * 5)))
                RMS = np.zeros_like(views_pre)
                for i, fov in enumerate(views_pre):
                    RMS[i] = self.calc_RMS(wavelength=wave_main_RGB[1], view=fov)
                RMS_diff = abs(np.diff(RMS))
                item1 = np.cumsum(RMS_diff) / RMS_diff.sum()
                item2 = np.linspace(0, 1, self.simulation.fov_num)
                views_pos = np.zeros_like(item2)
                for i, pos in enumerate(item2):
                    views_pos[i] = abs(item1 - pos).argmin()
                views_pos = views_pos / (len(item1) - 1)
                views = np.degrees(np.arctan(math.tan(math.radians(fov_max)) * views_pos))
            # 均匀采样用这个
            else:
                views = np.degrees(
                    np.arctan(math.tan(math.radians(fov_max)) * np.linspace(0, 1, self.simulation.fov_num)))
                views_pos = np.linspace(0, 1, self.simulation.fov_num)
            return views, views_pos

    # 在透镜前，计算得到近轴放大率，注意使用64位
    def calc_room_rate(self, wavelength=588.0, view=0.0):
        surfaces = self.basics.surfaces
        device = self.simulation.device
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
        ray = Ray(o, d, wavelength, device=device)
        # 输出光线，第一根为主光线，第二根为边缘光线
        _, ray_out = self._trace(ray, stop_ind=None, record=False)
        # 近轴放大率计算
        # 通过主光线求得近轴像位置
        o_out = ray_out.o[0]
        d_out = ray_out.d[0]
        t = (0 - o_out[0]) / d_out[0]
        d_sensor = o_out[2] + t * d_out[2]
        if self.basics.z_sensor is None:
            self.basics.z_sensor = d_sensor  # 更正位置
        # 通过近轴光线求得近轴像高度
        o_out = ray_out.o[1]
        d_out = ray_out.d[1]
        t = (d_sensor - o_out[2]) / d_out[2]
        img_r = o_out[0] + t * d_out[0]
        zoom_rate_pari = img_r / 1e-10
        return zoom_rate_pari.float()

    def calc_exit_pupil(self):
        # 孔阑中心反向追迹，得到入瞳位置
        surfaces = self.basics.surfaces
        device = self.simulation.device
        aperture_index = self.basics.aper_index
        wavelength = self.order1.wave_data["main"]
        if aperture_index >= len(surfaces) - 1:
            expp = surfaces[aperture_index].d
        else:
            o = torch.Tensor([1e-10, 0, surfaces[aperture_index + 1].d]).unsqueeze(0).to(device).double()
            d = torch.Tensor([1e-10, 0, surfaces[aperture_index + 1].d - surfaces[aperture_index].d]).unsqueeze(0).to(
                device).double()
            ray = Ray(o, d / d.square().sum().sqrt(), wavelength, device=device)
            valid, ray_out = self._trace(ray, record=False)
            if not valid[0]:
                raise Exception('ray trace is wrong!')
            o_out = ray_out.o[0]
            d_out = ray_out.d[0]
            expp = (o_out[2] + (0 - o_out[0]) / d_out[0] * d_out[2]).item()
        self.order1.expp = expp

    def resize_lens(self, wavelength=588.0, view=28.0, M=100001, vignetting=1, resume=True):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        aperture_index = self.basics.aper_index
        if resume:
            for i in range(len(surfaces)):
                if i != aperture_index:
                    surfaces[i].r = min((1 / ((1 + surfaces[i].k) * (surfaces[i].c ** 2)) ** 0.5).item(),
                                        self.basics.MAX_R)
        with torch.no_grad():
            angle = np.radians(view)
            R = surfaces[0].r  # [mm]
            sag = min(abs((np.tan(math.radians(view)) * surfaces[0].surface(surfaces[0].r, 0.0)).item()), R / 2)
            x = torch.linspace(-R - sag, R + sag, M, device=device)
            ones = torch.ones_like(x).to(device)
            zeros = torch.zeros_like(x).to(device)
            o = torch.stack((x, zeros, zeros))
            origin = torch.stack((-self.basics.z_object * ones * math.tan(angle), zeros, self.basics.z_object * ones))
            d = o - origin
            d = torch.div(d, d.norm(2, 0))
            ray = Ray(o.T, d.T, wavelength, device=device)
            valid, _, oss = self._trace(ray, record=True)
            if not valid.any():
                return False
            oss = oss[valid]
            aper_p = oss[:, aperture_index + 1, :]
            item1 = (aper_p[:, 0] - surfaces[aperture_index].r * vignetting).abs()
            item2 = (aper_p[:, 0] + surfaces[aperture_index].r * vignetting).abs()
            # if item1.min() <= 0.5 and item2.min() <= 0.5:
            index1 = torch.argmin(item1)
            index2 = torch.argmin(item2)
            p1 = oss[index1]
            p2 = oss[index2]
            for i in range(len(surfaces)):
                if i != aperture_index:
                    surfaces[i].r = max(abs(p1[i + 1, 0].item()), abs(p2[i + 1, 0].item())) + 0.1
            # else:
            #     return False
        return True

    def sample_ray_common(self, wavelength, view=0.0, M=15, R=None, sampling='grid', cen=None):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        angle = math.radians(view)  # 用弧度值代替度值数组
        if cen is None:
            cen = [0, 0]
        x = None
        y = None
        # R最大输入半径，即采样范围，这里为什么要加上倾斜的那一部分？应该是默认第一面作为孔径光阑
        if R is None:
            with torch.no_grad():
                # sag即为透镜边缘高度
                sag = surfaces[0].surface(surfaces[0].r, 0.0)
                R = np.tan(angle) * sag + surfaces[0].r  # [mm]
                R = R.item()
        if sampling == 'grid':
            x, y = torch.meshgrid(
                torch.linspace(-R + cen[0], R + cen[0], M, device=device),
                torch.linspace(-R + cen[1], R + cen[1], M, device=device),
                indexing='ij',
            )
        elif sampling == 'radial':
            r = torch.linspace(0, R, M, device=device)
            theta = torch.linspace(0, 2 * np.pi, M + 1, device=device)[0:M]
            x = cen[0] + r[None, ...] * torch.cos(theta[..., None])
            y = cen[1] + r[None, ...] * torch.sin(theta[..., None])
        # 假设第一个面z坐标为0，得到起点o
        ones = torch.ones_like(x, dtype=torch.float64)
        zeros = torch.zeros_like(x, dtype=torch.float64)
        o = torch.stack((x, y, zeros), dim=2).double()
        # 得到光线方向矢量d，2D方向改为3D方向，在x方向采样得到psf
        origin = torch.stack([(self.order1.enpp - self.basics.z_object) * math.tan(angle) * ones, zeros,
                              self.basics.z_object * ones], dim=2)
        d = o - origin
        d = torch.div(d, torch.stack((d.norm(2, 2), d.norm(2, 2), d.norm(2, 2)), dim=2))
        o = o.reshape(-1, 3)
        d = d.reshape(-1, 3)
        return Ray(o, d, wavelength, device=device)  # 这里返回的是一个光线类

    def sample_ray_2D_common(self, wavelength, view=0.0, M=15, R=None, cen=None):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        angle = math.radians(view)
        if cen is None:
            cen = [0, 0]
        # R最大输入半径，即采样范围，这里为什么要加上倾斜的那一部分？应该是默认第一面作为孔径光阑
        if R is None:
            with torch.no_grad():
                # sag即为透镜边缘高度
                sag = surfaces[0].surface(surfaces[0].r, 0.0)
                R = np.tan(angle) * sag + surfaces[0].r  # [mm]
                R = R.item()
        x = torch.linspace(-R + cen[0], R + cen[0], M, device=device)
        ones = torch.ones_like(x).to(device)
        zeros = torch.zeros_like(x).to(device)
        o = torch.stack((x, zeros, zeros))
        origin = torch.stack(
            ((self.order1.enpp - self.basics.z_object) * math.tan(angle) * ones, zeros, self.basics.z_object * ones))
        d = o - origin
        d = torch.div(d, d.norm(2, 0))
        return Ray(o.T, d.T, wavelength, device=device)

    # ------------------------------------------------------------------------------------
    def calc_rms(self, save_spot=False):
        num = self.diff.now_epochs
        fov_num = self.simulation.fov_num
        device = self.simulation.device
        wave_all_data = self.order1.wave_data["all"]
        self.order1.psf_cen_all = []
        self.order1.psf_R_all = []
        self.diff.loss_ray_edge = []
        for edof_num in range(len(self.basics.z_object_edof)):
            self.basics.z_object = self.basics.z_object_edof[edof_num]
            cen_sin = torch.zeros(fov_num, len(wave_all_data), 2).to(device)
            RMS_sin = torch.zeros(fov_num, len(wave_all_data)).to(device)
            for i, fov in enumerate(self.order1.fov_samples):
                for j, wave_len in enumerate(wave_all_data):
                    ray = self.sample_ray_common(wave_len, view=fov, R=self.order1.enp_xy[i][2],
                                                 M=self.simulation.enp_num,
                                                 sampling='grid', cen=self.order1.enp_xy[i][0:2])
                    valid, ray_final, oss = self._trace(ray, record=True)
                    t_img = (self.basics.z_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
                    p_img = ray_final(t_img)[valid]

                    d_final = ray_final.d[valid]
                    if i == len(self.order1.fov_samples) - 1:
                        self.diff.loss_ray_edge.append(
                            (p_img.transpose(0, 1) - (10 / d_final[..., 2]) * d_final.transpose(0, 1)).transpose(0, 1)[
                                ..., 0].abs().max())
                    # 通过RMS直径的1.5倍(经验值)作为psf范围
                    RMS, p_cen = self.rms(p_img, ceo=False)
                    cen_sin[i, j] = p_cen
                    RMS_sin[i, j] = RMS
                    # if save_main_psf and j == len(wave_all_data) // 2:
                    # if save_spot:
                    #     PSF_single = I.cpu().detach().numpy()
                    #     plt.figure(figsize=(24, 24))
                    #     plt.pcolormesh(PSF_single, cmap='magma', shading='flat')
                    #     plt.title("PSF_fov{fov}_wavelength{wave_len}".format(fov=round(fov.item(), 4),
                    #                                                          wave_len=round(wave_len)))
                    #     name = 'iter{num}_fov{fov}_wavelength{wave_len}'.format(fov=round(fov.item(), 4), num=num,
                    #                                                             wave_len=round(wave_len))
                    #     plt.savefig(self.path.demo_root + name + '.png')
                    #     plt.close()

            self.order1.psf_cen_all.append(cen_sin)
            self.order1.psf_R_all.append(RMS_sin)
        self.basics.z_object = self.basics.z_object_edof[len(self.basics.z_object_edof) // 2]
        return (torch.Tensor(self.order1.wave_data["all_weight"]).to(device) * (
                    sum(self.order1.psf_R_all) / len(self.order1.psf_R_all)).mean(dim=0)).sum() / 3

    def calc_psf(self, use_spot=True, save_main_psf=True, require_grad=False):
        num = self.diff.now_epochs
        fov_num = self.simulation.fov_num
        psf_num = self.simulation.psf_num
        device = self.simulation.device
        wave_all_data = self.order1.wave_data["all"]
        self.order1.raw_psf = []
        self.order1.PSF_real_pixel_size = []
        self.order1.d_PSF = []
        self.order1.psf_cen_all = []
        self.order1.psf_R_all = []
        self.diff.loss_ray_edge = []
        for edof_num in range(len(self.basics.z_object_edof)):
            self.basics.z_object = self.basics.z_object_edof[edof_num]
            PSF_sin = torch.zeros(fov_num, len(wave_all_data), psf_num, psf_num).to(device)
            PSF_real_pixel_size = np.zeros((fov_num, len(wave_all_data)))
            d_PSF_sin = None
            if require_grad:
                d_PSF_sin = torch.zeros(fov_num, len(wave_all_data), len(self.diff.parameters_all), psf_num,
                                        psf_num).to(device)
            psf_cen_sin = torch.zeros(fov_num, len(wave_all_data), 2).to(device)
            psf_R_sin = torch.zeros(fov_num, len(wave_all_data)).to(device)
            for i, fov in enumerate(self.order1.fov_samples):
                for j, wave_len in enumerate(wave_all_data):
                    ray = self.sample_ray_common(wave_len, view=fov, R=self.order1.enp_xy[i][2],
                                                 M=self.simulation.enp_num,
                                                 sampling='grid', cen=self.order1.enp_xy[i][0:2])
                    if i == len(self.order1.fov_samples) - 1:
                        calc_ray_edge = True
                    else:
                        calc_ray_edge = False
                    I, psf_world_pixel_size, d_I, RMS, p_cen = self.psf(ray, view=fov, wavelength=wave_len,
                                                                        use_spot=use_spot,
                                                                        require_grad=require_grad,
                                                                        calc_ray_edge=calc_ray_edge)
                    psf_cen_sin[i, j] = p_cen
                    psf_R_sin[i, j] = RMS
                    PSF_sin[i, j] = I
                    PSF_real_pixel_size[i, j] = psf_world_pixel_size
                    if require_grad:
                        d_PSF_sin[i, j] = d_I
                    # 保存主光线单色光PSF
                    # if save_main_psf and j == len(wave_all_data) // 2:
                    if save_main_psf and (edof_num == len(self.basics.z_object_edof) // 2):
                        PSF_single = I.cpu().detach().numpy()
                        plt.figure(figsize=(24, 24))
                        plt.pcolormesh(PSF_single, cmap='magma', shading='flat')
                        plt.title("PSF_fov{fov}_wavelength{wave_len}".format(fov=round(fov.item(), 4),
                                                                             wave_len=round(wave_len)))
                        name = 'iter{num}_fov{fov}_wavelength{wave_len}'.format(fov=round(fov.item(), 4), num=num,
                                                                                wave_len=round(wave_len))
                        plt.savefig(self.path.demo_root + name + '.png')
                        plt.close()

                    # # 将点扩散函数保存成.mat数据文件
                    # scipy.io.savemat('./out_compare/' + name + '.mat', PSF2)
                    # PSF2 = {'kernels': PSF_single}

                    # PSF尺寸统一，因为前面计算的各PSF物理尺寸不一样，需要进行统一化处理，先是把各个PSF尺寸统一，再将PSF和传感器像素尺寸统一
            self.order1.raw_psf.append(PSF_sin)
            self.order1.PSF_real_pixel_size.append(PSF_real_pixel_size)
            self.order1.d_PSF.append(d_PSF_sin)
            self.order1.psf_cen_all.append(psf_cen_sin)
            self.order1.psf_R_all.append(psf_R_sin)
            if require_grad:
                self.order1.raw_psf[edof_num].requires_grad = True
                self.order1.raw_psf[edof_num].grad = torch.zeros_like(self.order1.raw_psf[edof_num])
        self.basics.z_object = self.basics.z_object_edof[len(self.basics.z_object_edof) // 2]

    def uni_psf_pre(self, save_rgb_psf=True, count=0):
        num = self.diff.now_epochs
        PSF_uni = []
        for i in range(len(self.basics.z_object_edof)):
            PSF_uni.append(self.uni_psf(self.order1.raw_psf[i], self.order1.PSF_real_pixel_size[i]
                                        , self.order1.psf_cen_all[i], self.order1.psf_R_all[i]))

        self.order1.PSF = PSF_uni
        # if save_rgb_psf and count == 0:
        #     list_fov = [0, 7, -1]
        #     for j in range(len(PSF_uni)):
        #         for k in range(len(list_fov)):
        #             psf_draw = PSF_uni[j][list_fov[k]]
        #             # psf_R = (torch.Tensor(self.order1.exp_psf).reshape(len(PSF_uni), len(self.order1.fov_samples), len(self.order1.wave_data["all"]))[j][list_fov[k]]).mean()
        #             # 旋转图像90度，可以是90, 180, 270
        #             degrees = 135
        #             # 如果你想要旋转中心点，可以指定
        #             if k > 0:
        #                 psf_draw = transforms.functional.rotate(psf_draw, degrees, center=[20, 20], fill=[0, 0, 0])
        #             utils.save_image((torch.clamp(psf_draw, min=1e-10)) ** 0.5, self.path.demo_root + 'psf_rgb_epochs' + str(j) + '_' + str(k) + '.png',
        #                              nrow=3, normalize=True, padding=0)

    def psf(self, ray, view=torch.Tensor(0), wavelength=588, use_spot=True, require_grad=False, p_cen=None,
            RMS_rate=1, calc_ray_edge=False):
        """
        计算特定视场特定波长下的psf及其相应的梯度
        """
        # TODO: 提醒用户在使用此函数之前应该定义好传感器像素大小和分辨率大小。
        surfaces = self.basics.surfaces
        device = self.simulation.device
        psf_num = self.simulation.psf_num
        materials = self.basics.materials
        if self.order1.expp is None:
            exit_d = surfaces[-1].d
        else:
            exit_d = self.order1.expp
        # 光线追迹
        valid, ray_final, oss = self._trace(ray, record=True)
        oss = oss[valid]
        # 与传感器平面相交
        t_img = (self.basics.z_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
        p_img = ray_final(t_img)[valid]
        d_final = ray_final.d[valid]
        if calc_ray_edge:
            self.diff.loss_ray_edge.append(
                (p_img.transpose(0, 1) - (10 / d_final[..., 2]) * d_final.transpose(0, 1)).transpose(0, 1)[
                    ..., 0].abs().max())

        # 通过RMS直径的1.5倍(经验值)作为psf范围
        RMS, p_cen = self.rms(p_img, ceo=True)

        # 调试
        self.order1.exp_psf.append(self.rms(p_img, ceo=False)[0])

        p_cen = p_cen.float()
        RMS = RMS.float() * RMS_rate
        psf_world_pixel_size = 2 * RMS / psf_num

        # 点列图质心作为PSF半径
        x_main, y_main = p_cen[0], p_cen[1]
        x_img, y_img = torch.meshgrid(
            torch.linspace((x_main - RMS).item(), (x_main + RMS).item(), psf_num, device=device),
            torch.linspace(0, (y_main + RMS).item(), psf_num // 2 + 1, device=device),
            indexing='ij'
        )
        if use_spot:
            value_pos = p_img[:, :2].float()
            pos = (torch.stack([x_img, y_img], dim=2).reshape((-1, 2))).float()
            alpha = (psf_world_pixel_size * (2 ** 0.5) / 3).float()

            r = pos.unsqueeze(dim=1) - value_pos.unsqueeze(dim=0)

            PSF_Ixy0 = 1 / (alpha * (2 * torch.pi)) * torch.exp(-r.square().sum(dim=2) / (2 * (alpha ** 2))).sum(dim=1)
        else:
            # 与出瞳平面相交
            t_back = (exit_d - p_img[..., 2]) / d_final[..., 2]
            p_exit = p_img + torch.stack((t_back, t_back, t_back), dim=1) * d_final
            # 计算从物点到出瞳平面的光程
            angle = math.radians(view)
            origin = torch.zeros_like(p_img)  # origin为在物面的坐标
            origin[:, 0] = - self.basics.z_object * math.tan(angle)
            origin[:, 2] = self.basics.z_object
            material_list = torch.zeros(len(materials) + 1).to(device)
            for i in range(len(materials)):
                material_list[i] = materials[i].ior(wavelength)
            material_list[-1] = -materials[len(materials) - 1].ior(wavelength)
            opt_len_all, opt_len_rela = self.calc_opt_len(origin, oss, p_img, p_exit, material_list, is_far=False,
                                                          angle=angle)
            # 利用广播机制计算psf
            k = 2 * math.pi / (wavelength * 1e-6)  # 波数
            n = torch.Tensor([0, 0, 1]).to(device)  # 出瞳面法向量
            z_img = torch.ones_like(x_img) * self.basics.z_sensor
            # 像点坐标
            cor_img = torch.stack([x_img, y_img, z_img], dim=2).reshape([-1, 3])

            # 数据转换
            p_exit = p_exit.float()
            d_final = d_final.float()

            # 显存足够时可以用这个
            rx2y2 = cor_img.unsqueeze(dim=1) - p_exit.unsqueeze(dim=0)  # 广播机制
            rx2y2_len = torch.sqrt(torch.sum(torch.square(rx2y2), dim=2))
            K_factor = 0.5 * (torch.sum(n * rx2y2, dim=2) / rx2y2_len + torch.sum(n * d_final, dim=1))
            E = torch.sum(
                K_factor * (torch.exp(1j * k * opt_len_rela) / opt_len_all) * (
                        torch.exp(1j * k * rx2y2_len) / rx2y2_len),
                dim=1)
            PSF_Ixy0 = torch.real(E * torch.conj(E))
        PSF_Ixy = PSF_Ixy0 / PSF_Ixy0.max()
        PSF_Ixy1 = PSF_Ixy.reshape(psf_num, psf_num // 2 + 1)
        PSF_Ixy2 = torch.cat((PSF_Ixy1.flip(dims=[1])[:, :-1], PSF_Ixy1), dim=1)
        if require_grad:
            J = torch.zeros((len(self.diff.parameters_all)), len(PSF_Ixy)).to(device)
            # 雅克比矩阵
            v = torch.ones_like(PSF_Ixy, requires_grad=True)
            for i, x in enumerate(self.diff.parameters_all):
                N = torch.numel(x)
                #  view函数返回一个有相同数据但不同大小的 Tensor。通俗一点，就是改变矩阵维度
                vjp = torch.autograd.grad(PSF_Ixy, x, v, create_graph=True)[0].view(-1)

                if N == 1:
                    J[i, :] = torch.autograd.grad(vjp, v, retain_graph=False, create_graph=False)[0][
                        ..., None].squeeze(1)
                else:
                    I = torch.eye(N, device=x.device)
                    J = []
                    for ii in range(N):
                        Ji = torch.autograd.grad(vjp, v, I[ii], retain_graph=True)[0]
                        J.append(Ji.detach().clone())
                    J = torch.stack(J, dim=-1)
                del x.grad, v.grad
            J1 = J.reshape(len(self.diff.parameters_all), psf_num, psf_num // 2 + 1)
            J2 = torch.cat((J1.flip(dims=[2])[:, :, :-1], J1), dim=2)
        else:
            J2 = None
        return PSF_Ixy2.detach(), psf_world_pixel_size, J2, RMS, p_cen
        # 测试，查看点列图
        # with(torch.no_grad()):
        #     save_dir = './autodiff_demo/'
        #     ps = p_img[:, :2]
        #     item = 0.5
        #     name = 'fov{fov}_wavelength{wave_len}_RMS{rms}'.format(fov=round(view, 4), wave_len=round(wavelength), rms=round(RMS.item()/1.5*1000, 2))
        #     self.spot_diagram(ps.cpu(), x_lims=[-item, item], y_lims=[-item, item], save_path=save_dir + name + "spot-diagram.png"
        #                       , show=False)

    @staticmethod
    def calc_opt_len(origin, oss, p_img, p_exit, material_list, is_far=False, angle=0.0):
        origin = origin.unsqueeze(1)
        p_img = p_img.unsqueeze(1)
        p_exit = p_exit.unsqueeze(1)
        # oss这里可以删掉起始点
        p_all = torch.cat((origin, oss[:, 1:, :], p_img, p_exit), dim=1)
        item = torch.sqrt(torch.sum(p_all.diff(dim=1).square(), dim=2))
        opt_len_all = torch.sum(torch.mul(item, material_list), dim=1)

        direction = oss[:, 0, :].unsqueeze(1) - origin
        direction_len = torch.sqrt(torch.sum(torch.square(direction), dim=2))
        min_len = direction_len.min()
        direction = (direction.squeeze(1).T / direction_len.squeeze(1).T).T.unsqueeze(1)
        origin_rela = origin + min_len * direction
        # oss这里可以删掉起始点
        p_all = torch.cat((origin_rela, oss[:, 1:, :], p_img, p_exit), dim=1)
        item = torch.sqrt(torch.sum(p_all.diff(dim=1).square(), dim=2))
        opt_len_rela = torch.sum(torch.mul(item, material_list), dim=1)
        opt_len_rela = opt_len_rela - opt_len_rela.min()
        return opt_len_all.float(), opt_len_rela.float()

    def uni_psf(self, PSF, PSF_real_pixel_size, psf_cen_all, psf_R_all):
        device = self.simulation.device
        wave_weight = self.order1.wave_data["all_weight"]
        psf_pixel_size = self.simulation.img_pixel_size
        # 确定PSF最大范围
        D_max = ((psf_cen_all[:, :, 0] + psf_R_all).max(dim=1)[0] - (psf_cen_all[:, :, 0] - psf_R_all).min(dim=1)[
            0]).max().detach()
        psf_final_num = int(D_max / psf_pixel_size)
        if psf_final_num >= self.simulation.psf_num_max:
            psf_final_num = self.simulation.psf_num_max
        if psf_final_num % 2 == 0:
            psf_final_num = psf_final_num + 1
        stan_num = self.simulation.psf_num
        PSF_uni = torch.zeros(PSF.size(0), PSF.size(1), psf_final_num, psf_final_num).to(device)
        PSF_final = torch.zeros(PSF.size(0), 3, psf_final_num, psf_final_num).to(device)
        for i in range(PSF.size(0)):
            p_cen_single_fov = psf_cen_all[i, PSF.size(1) // 2].detach()
            for j in range(PSF.size(1)):
                PSF_single = PSF[i, j]
                rate = PSF_real_pixel_size[i, j] / psf_pixel_size
                psf_num = int(stan_num * rate)
                if psf_num % 2 == 0:
                    psf_num = psf_num + 1
                psf_single1 = resize(PSF_single.unsqueeze(0).unsqueeze(0), [psf_num, psf_num],
                                     interpolation=InterpolationMode.BICUBIC)
                p_cen = psf_cen_all[i, j].detach()
                pad_y = (psf_final_num - psf_num) // 2
                pixel_bias = int(((p_cen - p_cen_single_fov) / psf_pixel_size)[0])
                pad_x_up = pad_y - pixel_bias
                pad_x_down = pad_y + pixel_bias
                psf_single2 = F.pad(psf_single1, (pad_y, pad_y, pad_x_up, pad_x_down), mode="constant", value=0)
                PSF_uni[i, j] = psf_single2 / psf_single2.sum() * wave_weight[j]
        wave_num = PSF.size(1) // 3
        for channel in range(3):
            PSF_final[:, channel] = PSF_uni[:, channel * wave_num:(channel + 1) * wave_num].sum(dim=1)
        return PSF_final

    # 计算weight_mask，用在损失上
    def calc_loss_map(self, save=False):
        with torch.no_grad():
            device = self.simulation.device
            N, C, H, W = self.simulation.b_size, 3, self.simulation.img_size[0], self.simulation.img_size[1]
            interval_num = int(((H / 2) ** 2 + (W / 2) ** 2) ** 0.5) + 1
            wave_num = self.order1.psf_R_all.size(1)
            RMS = self.order1.psf_R_all[:, wave_num // 2]
            loss_weight_list = F.interpolate(RMS.unsqueeze(0).unsqueeze(0),
                                             size=interval_num, mode='linear',
                                             align_corners=True).squeeze(0).squeeze(0)
            x, y = torch.meshgrid(
                torch.linspace(-H // 2, H // 2, H, device=device),
                torch.linspace(-W // 2, W // 2, W, device=device),
                indexing='ij',
            )
            loss_map = loss_weight_list[
                           torch.sqrt(torch.square(x) + torch.square(y)).long()] ** self.simulation.loss_map_weight
            loss_map = loss_map / loss_map.min()
            if save:
                utils.save_image(loss_map / loss_map.max(),
                                 self.path.demo_root + '{iter}_loss_map.png'.format(iter=self.simulation.now_epochs))
        return loss_map

    def calc_main_pos(self, wavelength, view=0.0, M=10000, R=None, cen=None):
        ray = self.sample_ray_2D_common(wavelength, view=view, M=M, R=R, cen=cen)
        ray_final, valid, oss = self.trace_r(ray)
        oss = oss[valid]
        # 与传感器平面相交
        t_img = (self.d_sensor - ray_final.o[..., 2]) / ray_final.d[..., 2]
        p_img = ray_final(t_img)[valid]
        aper_index = 0
        for i, s in enumerate(self.surfaces):
            if (i + 1) < len(self.surfaces) and self.materials[i].A < 1.0003 and self.materials[i + 1].A < 1.0003:
                aper_index = i
                break
        aper_p = oss[:, aper_index + 1, :]
        main_index = torch.argmin(aper_p[:, 0].abs())

        return p_img[main_index, 0:2]

    def calc_loss_constraint(self, e=4, weight_bfl=1, weight_material=1):
        surfaces = self.basics.surfaces
        device = self.simulation.device
        materials = self.basics.materials
        is_normal = True
        loss_limits = []
        # 透镜间隔和材料的物理约束
        for i in range(len(surfaces) - 1):
            d_cen = surfaces[i + 1].d - surfaces[i].d
            sag2 = surfaces[i + 1].surface(surfaces[i + 1].r, 0.0)
            sag1 = surfaces[i].surface(surfaces[i].r, 0.0)
            d_border = (surfaces[i + 1].d + sag2) - (surfaces[i].d + sag1)
            # 玻璃中心和边缘厚度 + 材料约束
            if (materials[i].n < 1.0003 < materials[i + 1].n) or (
                    materials[i].n > 1.0003 and materials[i + 1].n > 1.0003):
                # loss_limits.append(self.loss_constraint(d_cen, self.constraint.thi_glass_cen, exponent=e))
                loss_limits.append(
                    self.loss_constraint(d_border, self.constraint.thi_glass_border, exponent=e, zero_flag=True))

                acc_idx = (materials[i + 1].n_all < self.constraint.material_n[1]) & (materials[i + 1].n_all > self.constraint.material_n[0]) & (materials[i + 1].V_all < self.constraint.material_V[1]) & (materials[i + 1].V_all > self.constraint.material_V[0])

                loss_material = ((((materials[i + 1].n - materials[i + 1].n_all[acc_idx]) / 0.1) ** 2 +
                                 ((materials[i + 1].V - materials[i + 1].V_all[acc_idx]) / 50) ** 2)).min()

                if loss_material > 0:
                    loss_limits.append(loss_material * weight_material)
                # loss_limits.append(self.loss_constraint(materials[i + 1].V, self.constraint.material_V, exponent=e))
            # 空气中心和边缘厚度
            elif 0 <= self.basics.aper_index - i <= 1:
                # loss_limits.append(self.loss_constraint(d_cen, self.constraint.aper_d, exponent=e))
                loss_limits.append(self.loss_constraint(d_border, self.constraint.aper_d, exponent=e, zero_flag=True))
            else:
                # loss_limits.append(self.loss_constraint(d_cen, self.constraint.thi_air_cen, exponent=e))
                loss_limits.append(
                    self.loss_constraint(d_border, self.constraint.thi_air_border, exponent=e, zero_flag=True))
        # 最后一面和像面距离，后截距
        sag1 = max(surfaces[-1].surface(surfaces[-1].r, 0.0), 0)
        bfl = self.basics.z_sensor - (surfaces[-1].d + sag1)
        loss_limits.append(self.loss_constraint(bfl, self.constraint.bfl, exponent=e, zero_flag=True) * weight_bfl)

        # 像面约束
        loss_distort = torch.zeros(1)[0].to(device)
        for i in range(len(self.basics.z_object_edof)):
            for j in range(len(self.order1.fov_pos)):
                if 0 < j < len(self.order1.fov_pos)-1:
                    pos_real = self.order1.psf_cen_all[i][j, :, 0].mean()
                    pos_ideal = self.order1.img_r * self.order1.fov_pos[j]
                    loss_distort = loss_distort + self.loss_constraint((pos_ideal + pos_real) / pos_ideal,
                                                                       self.constraint.distort, exponent=e,
                                                                       zero_flag=True)
                elif j == len(self.order1.fov_pos) - 1:
                    pos_real = self.order1.psf_cen_all[i][j, :, 0].mean()
                    pos_ideal = self.order1.img_r * self.order1.fov_pos[j]
                    loss_distort = loss_distort + self.loss_constraint((pos_ideal + pos_real) / pos_ideal,
                                                                       self.constraint.img, exponent=e,
                                                                       zero_flag=False)
        # pos_img = (self.order1.psf_cen_all[len(self.basics.z_object_edof) // 2][-1][..., 0].mean()).abs()
        loss_limits.append(loss_distort)

        # 焦距
        effl_diff = self.calc_effl(wavelength=self.order1.wave_data["main"])
        loss_limits.append(self.loss_constraint(effl_diff, self.constraint.effl, exponent=e, zero_flag=True))

        # 边缘光线
        # loss_ray_edge = (sum(self.diff.loss_ray_edge) / len(self.diff.loss_ray_edge) - self.constraint.edge_ray) / 0.3
        # if loss_ray_edge <= 0:
        #     loss_ray_edge = 0
        # loss_limits.append(loss_ray_edge ** e)

        res = sum(loss_limits)
        # if res <= 0:
        #     res = torch.Tensor([res])[0].to(device)
        return res

    @staticmethod
    def loss_constraint(x, bound, exponent=10, zero_flag=True):
        min_x = bound[0]
        max_x = bound[1]

        # 指数型，梯度容易爆炸，慎用
        # h = (max_x - min_x) * scale_pos
        # b = scale ** (-1 / h)
        # loss_con = b**(x-max_x) + b**(min_x-x)

        # 幂函数型，幂指数设置为10
        cen = (max_x + min_x) / 2
        diff = (max_x - min_x) / 2

        # else:
        # loss_con = (((x - cen).abs() / diff) ** exponent)

        if zero_flag:
            loss_con = (((x - cen).abs() / diff) ** exponent - 1)
            if min_x <= x <= max_x:
                loss_con = loss_con.abs() * 1e-10
        else:
            loss_con = (((x - cen).abs() / diff) ** exponent)
        # h = (max_x - min_x) * scale_pos
        # y = np.linspace(min_x-h, max_x+h, 1000)
        # loss = a*(y-cen)**exponent
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.plot(y, loss)
        # plt.xlabel('x ')
        # plt.ylabel('loss')
        # fig.savefig('./autodiff_demo/loss_constraint.png', bbox_inches='tight')
        # y = np.linspace(min_x-h, max_x+h, 1000)
        # loss = b ** (y - max_x) + b ** (min_x - y)
        # fig = plt.figure()
        # ax = plt.axes()
        # ax.plot(y, loss)
        # plt.xlabel('x ')
        # plt.ylabel('loss')
        # fig.savefig('./autodiff_demo/loss_constraint.png', bbox_inches='tight')

        return loss_con

    def write_optics_data(self, save_path='./end2end_0708.txt', round_num=5, loss=None, end_flag=False):
        surfaces = self.basics.surfaces
        materials = self.basics.materials
        iteration = self.simulation.now_epochs

        with open(save_path, 'a', encoding='utf-8') as f:  # 使用with open()新建对象f
            if end_flag:
                f.write('final_system:' + '\n')
                f.write('selected_materials:' + '\n')
                for j in range(len(self.diff.mater_name)):
                    f.write(self.diff.mater_name[j] + '\n')
                # f.write('selected_test_plate:' + '\n')
                # for j in range(len(self.diff.test_plate_order)):
                #     f.write(self.diff.test_plate_order[j] + '\n')
            f.write('num:' + str(iteration) + '\n')
            item = "order   radius     thickness     n     V     r     conic     ai      loss={x}    aper_idx={y}\n".format(x=loss, y=self.basics.aper_index)
            for i in range(len(surfaces)):
                if i < len(surfaces) - 1:
                    thi = surfaces[i + 1].d.item() - surfaces[i].d.item()
                else:
                    thi = self.basics.z_sensor.item() - surfaces[i].d.item()
                if i == self.basics.aper_index:
                    c_r = math.inf
                else:
                    c_r = round(1 / surfaces[i].c.item(), round_num)
                item += "{num}      {c_r}       {thi}       {n}       {V}       {r}       {k}      " \
                    .format(num=i, c_r=c_r, thi=round(thi, round_num),
                            k=round(surfaces[i].k.item(), round_num),
                            r=round(surfaces[i].r, round_num),
                            n=round(materials[i + 1].n.item(), round_num),
                            V=round(materials[i + 1].V.item(), round_num))
                if surfaces[i].ai is not None:
                    for ii in range(len(surfaces[i].ai)):
                        item += "{ai} ".format(ai=surfaces[i].ai[ii].item())
                item += "\n"
            item += 'end\n\n'
            f.write(item)

    @staticmethod
    def read_data(read_path, device, now_epochs=8):
        surfaces = []
        materials = [Material(device=device, name='air')]
        with open(read_path, 'r', encoding='utf-8') as f:  # 使用with open()新建对象f
            contents = f.readlines()
            for i in range(len(contents)):
                if contents[i] == 'num:{num}\n'.format(num=now_epochs):
                    d_now = 0
                    # 孔阑位置
                    aperture_index = int(str.split(contents[i + 1])[-1][9:])
                    while contents[i + 2] != 'end\n':
                        content = str.split(contents[i + 2])
                        if int(content[0]) > 0:
                            d_now = d_now + float(str.split(contents[i + 1])[2])
                        if len(content) <= 7:
                            surfaces.append(
                                Aspheric(float(content[5]), d_now, c=1 / float(content[1]), k=float(content[6]),
                                         device=device))
                        else:
                            ai = []
                            for ii in range(7, len(content)):
                                ai.append(torch.Tensor([float(content[ii])])[0].to(device))
                            surfaces.append(
                                Aspheric(float(content[5]), d_now, c=1 / float(content[1]), k=float(content[6]),
                                         ai=ai, device=device))
                        materials.append(Material(device=device, data=[float(content[3]), float(content[4])]))
                        i += 1
                    z_sensor = torch.Tensor([d_now + float(str.split(contents[i + 1])[2])])[0].to(device)
                    break
        f.close()
        return surfaces, materials, aperture_index, z_sensor

    def trace_valid(self, ray):
        """
        追迹光线以查看它们是否与传感器平面相交。
        """
        valid = self._trace(ray)[1]
        return valid

    def trace_to_sensor(self, ray, record=False):
        """
        光线追迹，使其与传感器平面相交。
        """
        valid, ray_out, oss = self._trace(ray, record=True)
        # 与传感器平面相交
        t = (self.basics.z_sensor - ray_out.o[..., 2]) / ray_out.d[..., 2]
        p = ray_out(t)
        p = p[valid]
        if record:
            oss = oss[valid]
            return oss, p
        else:
            return p

    @staticmethod
    def _refract(wi, n, eta, approx=False):
        """
        Snell's law (surface normal n defined along the positive z axis)
        https://physics.stackexchange.com/a/436252/104805
        """
        if type(eta) is float:
            eta_ = eta
        else:
            if np.prod(eta.shape) > 1:
                eta_ = eta[..., None]
            else:
                eta_ = eta

        cosi = torch.sum(wi * n, dim=-1)

        if approx:
            tmp = 1. - eta ** 2 * (1. - cosi)
            valid = tmp > 0.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        else:
            cost2 = 1. - (1. - cosi ** 2) * eta ** 2

            # 1. get valid map; 2. zero out invalid points; 3. add eps to avoid NaN grad at cost2==0.
            valid = cost2 > 0.
            cost2 = torch.clamp(cost2, min=1e-8)
            tmp = torch.sqrt(cost2)

            # here we do not have to do normalization because if both wi and n are normalized,
            # then output is also normalized.
            wt = tmp[..., None] * n + eta_ * (wi - cosi[..., None] * n)
        return valid, wt

    def _trace(self, ray, stop_ind=None, record=False):
        if stop_ind is None:
            stop_ind = len(self.basics.surfaces) - 1  # 最后一面作为stop
        is_forward = (ray.d[..., 2] > 0).all()  # 确认光线是否正向传播
        if is_forward:
            return self._forward_tracing(ray, stop_ind, record)
        else:
            return self._backward_tracing(ray, stop_ind, record)

    def _forward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        surfaces = self.basics.surfaces
        materials = self.basics.materials
        device = self.simulation.device
        oss = torch.ones(ray.o.shape[0], stop_ind + 2, ray.o.shape[1]).to(device)
        if record:
            oss[:, 0, :] = ray.o
        valid = torch.ones(ray.o.shape[0], device=device).bool()
        for i in range(stop_ind + 1):
            # 两个表面折射率之比
            eta = materials[i].ior(wavelength) / materials[i + 1].ior(wavelength)
            # 光线与透镜表面相交，p为与透镜表面交点，
            valid_o, p = surfaces[i].ray_surface_intersection(ray, valid)
            valid = valid & valid_o
            if not valid.any():
                break
            item2 = p[valid]
            # 异常值范围限制，防止计算报错
            p2 = torch.stack((torch.clamp(p[:, 0], min=item2[:, 0].min().item(), max=item2[:, 0].max().item()),
                              torch.clamp(p[:, 1], min=item2[:, 1].min().item(), max=item2[:, 1].max().item()),
                              torch.clamp(p[:, 2], min=item2[:, 2].min().item(), max=item2[:, 2].max().item())), dim=1)
            # 得到透镜表面法线
            n = surfaces[i].normal(p2[..., 0], p2[..., 1])
            if self.diff.loss_raytrace_flag:
                # cos_total_reflection = min(torch.cos(torch.arcsin(1 / eta)), torch.Tensor([1])[0].to(self.device))
                # cos_total_reflection = torch.cos(torch.arcsin(eta))
                item = (1 + ((ray.d * n)[valid & valid_o].sum(dim=1)))
                # item = 1000**(-((-ray.d * n)[valid].sum(dim=1)-cos_total_reflection))
                self.diff.loss_raytrace = self.diff.loss_raytrace + item.mean()
                self.diff.loss_raytrace_num += 1
            valid_d, d = self._refract(ray.d, -n, eta)
            # 检验有效性
            valid = valid & valid_d
            if not valid.any():
                break
            ray.o = p2
            ray.d = d
            # 更新光线 {o,d}
            if record:  # TODO: make it pythonic ...
                oss[:, i + 1, :] = ray.o
        if record:
            return valid, ray, oss
        else:
            return valid, ray

    def _backward_tracing(self, ray, stop_ind, record):
        wavelength = ray.wavelength
        surfaces = self.basics.surfaces
        materials = self.basics.materials
        device = self.simulation.device
        dim = ray.o[..., 2].shape
        oss = torch.ones(ray.o.shape[0], stop_ind + 2, ray.o.shape[1]).to(device)
        if record:
            oss[:, -1, :] = ray.o
        valid = torch.ones(dim, device=ray.o.device).bool()
        for i in np.flip(range(stop_ind + 1)):
            surface = surfaces[i]
            eta = materials[i + 1].ior(wavelength) / materials[i].ior(wavelength)
            # ray intersecting surface
            valid_o, p = surface.ray_surface_intersection(ray, valid)

            valid = valid & valid_o
            if not valid.any():
                break

            item2 = p[valid]
            # 异常值范围限制，防止计算报错
            p2 = torch.stack((torch.clamp(p[:, 0], min=item2[:, 0].min().item(), max=item2[:, 0].max().item()),
                              torch.clamp(p[:, 1], min=item2[:, 1].min().item(), max=item2[:, 1].max().item()),
                              torch.clamp(p[:, 2], min=item2[:, 2].min().item(), max=item2[:, 2].max().item())), dim=1)

            # get surface normal and refract 
            n = surface.normal(p2[..., 0], p2[..., 1])

            # if eta > 1 and self.flag_calc_loss_reg:
            if self.diff.loss_raytrace_flag:
                # xx = min(1/eta, torch.Tensor([1])[0].to(self.device))
                # cos_total_reflection = min(torch.cos(torch.arcsin(1/eta)), torch.Tensor([1])[0].to(self.device))
                item = 1000 ** (-(ray.d * n)[valid & valid_o].sum(dim=1))
                self.diff.loss_raytrace = self.diff.loss_raytrace + item.mean()
                self.diff.loss_raytrace_num += 1
            valid_d, d = self._refract(ray.d, n, eta)  # backward: no need to revert the normal
            # check validity
            valid = valid & valid_d
            if not valid.any():
                break
            # update ray {o,d}
            ray.o = p2
            ray.d = d
            if record:
                oss[:, i, :] = ray.o
        if record:
            return valid, ray, oss
        else:
            return valid, ray


class Surface(PrettyPrinter):
    def __init__(self, r, d, is_square=False, device=torch.device('cpu')):
        # self.r = torch.Tensor(np.array(r))
        self.c = None
        self.d = None
        if torch.is_tensor(d):
            self.d = d
        else:
            self.d = torch.Tensor(np.asarray(float(d))).to(device)
        self.is_square = is_square
        self.r = float(r)
        self.device = device
        self.z_con = None

        # 控制光线追迹精度的参数
        self.NEWTONS_MAX_ITER = 50
        self.NEWTONS_TOLERANCE_TIGHT = 50e-8  # in [mm], i.e. 50 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 300e-8  # in [mm], i.e. 300 [nm] here (up to <10 [nm])
        self.APERTURE_SAMPLING = 257

    # === 一般方法 (一定不能被覆盖的)
    def surface_with_offset(self, x, y):
        return self.surface(x, y) + self.d

    def normal(self, x, y):
        ds_dxyz = self.surface_derivatives(x, y)
        return normalize(torch.stack(ds_dxyz, dim=-1))

    def surface_area(self):
        if self.is_square:
            return self.r ** 2
        else:  # is round
            return math.pi * self.r ** 2

    def mesh(self):
        """
        为当前曲面生成网格。
        """
        x, y = torch.meshgrid(
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            torch.linspace(-self.r, self.r, self.APERTURE_SAMPLING, device=self.device),
            indexing='ij'
        )
        valid_map = self.is_valid(torch.stack((x, y), axis=-1))
        return self.surface(x, y) * valid_map

    def sdf_approx(self, p):  # 近似的 SDF
        """
        This function is more computationally efficient than `sdf`.
        - < 0: valid
        """
        if self.is_square:
            return torch.max(torch.abs(p) - self.r, dim=-1)[0]
        else:  # is round
            return length2(p) - self.r ** 2

    def is_valid(self, p):
        return (self.sdf_approx(p) < 0.0).bool()

    def ray_surface_intersection(self, ray, active=None):
        """
        Returns:
        - p: 交点
        - g: 显式函数
        """
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)

        valid_o = solution_found & self.is_valid(local[..., 0:2])
        if active is not None:
            valid_o = active & valid_o
        return valid_o, local

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        if oz.numel() < 2:
            oz = torch.Tensor([oz.item()]).to(self.device)
        t_delta = torch.zeros_like(oz)

        # 迭代直到相交误差很小
        t = maxt * torch.ones_like(oz)
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAX_ITER):
            it += 1
            t = t0 + t_delta  # 相加的t0即是在透镜顶点的倍数，t_delta是后面的
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz + self.z_con, A, B, C  # here z = t_delta * dz
            )
            s_derivatives_dot_D[s_derivatives_dot_D.abs() < 1e-12] = 1e-12
            s_derivatives_dot_D[s_derivatives_dot_D.abs() > 1e24] = 1e24
            residual[residual.abs() < 1e-24] = 1e-24
            residual[residual.abs() > 1e12] = 1e12

            t_delta = t_delta - residual / s_derivatives_dot_D
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    def newtons_method(self, maxt, o, D, option='implicit'):
        # 牛顿法求光线曲面交点方程的根。
        # 支持两种模式:
        # 
        # 1. 'explicit": 利用 autodiff 实现循环, 而且梯度对于 o, D, and self.parameters 是准确的. 不仅缓慢而且耗费内存.
        # 
        # 2. 'implicit": 这使用隐式层理论实现了循环, 不用autodiff, 然后勾选梯度。更少的内存消耗。

        # 计算前常数
        ox, oy, oz = (o[..., i].clone() for i in range(3))
        dx, dy, dz = (D[..., i].clone() for i in range(3))
        A = torch.clamp(dx ** 2 + dy ** 2, min=1e-8)
        B = 2 * (dx * ox + dy * oy)
        C = ox ** 2 + oy ** 2

        # initial guess of t
        t0 = (self.d - oz) / dz
        if self.c.abs() > 1e-3:
            item_t0 = (1 / (self.c * (1 + self.k) ** 0.5)).abs() - C ** 0.5
            t0[item_t0 <= 0] = ((item_t0 / (A ** 0.5) * (1 + 1e-2))[item_t0 <= 0]).abs()
        self.z_con = (t0 - (self.d - oz) / dz) * dz

        if option == 'explicit':
            t, t_delta, valid = self.newtons_method_impl(
                maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
            )
        elif option == 'implicit':
            with torch.no_grad():
                t, t_delta, valid = self.newtons_method_impl(
                    maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C
                )
            s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz + self.z_con, A, B, C
            )[1]
            t = t0 + t_delta
            residual = (self.g(ox + t * dx, oy + t * dy) + self.h(oz + t * dz) + self.d)
            s_derivatives_dot_D[s_derivatives_dot_D.abs() < 1e-12] = 1e-12
            s_derivatives_dot_D[s_derivatives_dot_D.abs() > 1e24] = 1e24
            residual[residual.abs() < 1e-24] = 1e-24
            residual[residual.abs() > 1e12] = 1e12
            t = t - residual / s_derivatives_dot_D
        else:
            raise Exception('option={} is not available!'.format(option))

        p = o + t[..., None] * D
        return valid, p

    # === Virtual methods (must be overridden)
    def g(self, x, y):
        raise NotImplementedError()

    def dgd(self, x, y):
        """
        Derivatives of g: (g'x, g'y).
        """
        raise NotImplementedError()

    def h(self, z):
        raise NotImplementedError()

    def dhd(self, z):
        """
        Derivative of h.
        """
        raise NotImplementedError()

    def surface(self, x, y):
        """
        Solve z from h(z) = -g(x,y).
        """
        raise NotImplementedError()

    def reverse(self):
        raise NotImplementedError()

    # === 默认方法 (最好被覆盖)
    def surface_derivatives(self, x, y):
        """
        Returns \nabla f = \nabla (g(x,y) + h(z)) = (dg/dx, dg/dy, dh/dz).
        (Note: this default implementation is not efficient)
        """
        gx, gy = self.dgd(x, y)
        z = self.surface(x, y)
        return gx, gy, self.dhd(z)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        """
        Returns g(x,y)+h(z) and dot((g'x,g'y,h'), (dx,dy,dz)).
        (Note: this default implementation is not efficient)
        """
        x = ox + t * dx
        y = oy + t * dy
        s = self.g(x, y) + self.h(z)
        sx, sy = self.dgd(x, y)
        sz = self.dhd(z)
        return s, sx * dx + sy * dy + sz * dz


class Aspheric(Surface):
    """
    非球面: https://en.wikipedia.org/wiki/Aspheric_lens.
    c：曲率，曲率半径的导数
    k：二次圆锥系数
    ai：非球面系数
    """

    def __init__(self, r, d, c=0., k=0., ai=None, is_square=False, device=torch.device('cpu')):
        Surface.__init__(self, r, d, is_square, device)
        self.c, self.k = (torch.Tensor(np.array(v)).to(device) for v in [c, k])
        self.ai = None
        if ai is not None:
            self.ai = ai

    # === Common methods
    def g(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def dgd(self, x, y):
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y

    def h(self, z):
        return -z

    def dhd(self, z):
        return -torch.ones_like(z)

    def surface(self, x, y):
        return self._g(x ** 2 + y ** 2)

    def reverse(self):
        self.c = -self.c
        if self.ai is not None:
            self.ai = -self.ai

    def surface_derivatives(self, x, y):
        dsdr2 = 2 * self._dgd(x ** 2 + y ** 2)
        return dsdr2 * x, dsdr2 * y, -torch.ones_like(x)

    def surface_and_derivatives_dot_D(self, t, dx, dy, dz, ox, oy, z, A, B, C):
        # pylint: disable=unused-argument
        # TODO: could be further optimized
        r2 = A * t ** 2 + B * t + C  # 加上t*d后x^2+y^2的值
        return self._g(r2) - z, self._dgd(r2) * (2 * A * t + B) - dz  # D与梯度的点乘积

    # === Private methods
    # 已知x,y坐标求z函数的准确值
    def _g(self, r2):
        tmp = r2 * self.c
        item = 1 - (1 + self.k) * tmp * self.c
        item = torch.clamp(item, min=1e-8)
        total_surface = tmp / (1 + torch.sqrt(item))
        higher_surface = 0
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_surface = r2 * higher_surface + self.ai[i]
            higher_surface = higher_surface * r2 ** 2
        return total_surface + higher_surface

    # 求相应梯度
    def _dgd(self, r2):
        alpha_r2 = (1 + self.k) * self.c ** 2 * r2  # k为圆锥系数，c为曲率，r2为x^2+y^2
        item = 1 - alpha_r2
        item = torch.clamp(item, min=1e-8)
        tmp = torch.sqrt(item)  # TODO: potential NaN grad
        total_derivative = self.c * (1 + tmp - 0.5 * alpha_r2) / (tmp * (1 + tmp) ** 2)  # 基本的梯度

        higher_derivative = 0  # 高次项梯度
        if self.ai is not None:
            for i in np.flip(range(len(self.ai))):
                higher_derivative = r2 * higher_derivative + (i + 2) * self.ai[i]
        return total_derivative + higher_derivative * r2
