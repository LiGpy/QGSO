import torch
import math
import numpy as np
import pandas as pd
import time

class Surfaces:
    def __init__(self, r, d):
        device = d.device
        self.device = device
        self.z_con = None
        self.d = d
        self.r = r
        self.c = None
        self.k = None
        self.ai = None

        # 控制光线追迹精度的参数
        self.NEWTONS_MAX_ITER = 5
        self.NEWTONS_TOLERANCE_TIGHT = 10e-6  # in [mm], i.e. 10 [nm] here (up to <10 [nm])
        self.NEWTONS_TOLERANCE_LOOSE = 100e-6  # in [mm], i.e. 100 [nm] here (up to <10 [nm])
        self.APERTURE_SAMPLING = 63

    # === 一般方法 (一定不能被覆盖的)
    def surface_with_offset(self, x, y):
        return self.surface(x, y) + self.d

    def normal(self, x, y):
        ds_dxyz = self.surface_derivatives(x, y)
        d = torch.stack(ds_dxyz, dim=0)
        return d / d.square().sum(dim=0).sqrt()

    def surface_area(self):
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
        valid_map = self.is_valid(torch.stack((x, y), dim=-1))
        return self.surface(x, y) * valid_map

    def sdf_approx(self, p):  # 近似的 SDF
        return (p ** 2).sum(dim=0) - self.r ** 2

    def is_valid(self, p):
        return (self.sdf_approx(p) < 0.0).bool()

    def ray_surface_intersection(self, ray, active=None):
        """
        Returns:
        - p: 交点
        - g: 显式函数
        """
        solution_found, local = self.newtons_method(ray.maxt, ray.o, ray.d)

        valid_o = solution_found & self.is_valid(local[0:2])
        if active is not None:
            valid_o = active & valid_o
        return valid_o, local

    def newtons_method_impl(self, maxt, t0, dx, dy, dz, ox, oy, oz, A, B, C):
        t_delta = torch.zeros_like(oz)
        # 迭代直到相交误差很小
        residual = maxt * torch.ones_like(oz)
        it = 0
        while (torch.abs(residual) > self.NEWTONS_TOLERANCE_TIGHT).any() and (it < self.NEWTONS_MAX_ITER):
            it += 1
            t = t0 + t_delta  # 相加的t0即是在透镜顶点的倍数，t_delta是后面的
            residual, s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
                t, dx, dy, dz, ox, oy, t_delta * dz + self.z_con, A, B, C  # here z = t_delta * dz
            )
            s_derivatives_dot_D[s_derivatives_dot_D == 0] = 1e-12
            # s_derivatives_dot_D[s_derivatives_dot_D.abs() < 1e-12] = 1e-12
            # s_derivatives_dot_D[s_derivatives_dot_D.abs() > 1e24] = 1e24
            # residual[residual.abs() < 1e-24] = 1e-24
            # residual[residual.abs() > 1e12] = 1e12

            t_delta = t_delta - residual / s_derivatives_dot_D
        t = t0 + t_delta
        valid = (torch.abs(residual) < self.NEWTONS_TOLERANCE_LOOSE) & (t <= maxt)
        return t, t_delta, valid

    def newtons_method(self, maxt, o, D, option='explicit'):
        # 牛顿法求光线曲面交点方程的根。
        # 支持两种模式:
        #
        # 1. 'explicit": 利用 autodiff 实现循环, 而且梯度对于 o, D, and self.parameters 是准确的. 不仅缓慢而且耗费内存.
        #
        # 2. 'implicit": 这使用隐式层理论实现了循环, 不用autodiff, 然后勾选梯度。更少的内存消耗，遗传算法不涉及梯度所以暂不使用。

        # 计算前常数
        # time1 = time.time()
        ox, oy, oz = (o[i].clone() for i in range(3))
        dx, dy, dz = (D[i].clone() for i in range(3))
        A = dx ** 2 + dy ** 2
        B = 2 * (dx * ox + dy * oy)
        C = ox ** 2 + oy ** 2
        # 对t的初始定位
        t0 = (self.d - oz) / dz

        with torch.no_grad():
            item_rate = 0.5
            t_init = torch.zeros_like(t0)
            s_old = self.g(ox, oy) + self.d - oz
            item_t0 = self.r - (ox ** 2 + oy ** 2) ** 0.5
            # 通过逼近法先求解一个合适的t
            s_old[item_t0 <= 0] = 1
            rate = item_rate * t0
            it = 0
            while (torch.abs(s_old) > self.NEWTONS_TOLERANCE_LOOSE).any() and (it < 3):
                it += 1  # 相加的t0即是在透镜顶点的倍数，t_delta是后面的
                t_init = t_init + rate
                x = ox + t_init * dx
                y = oy + t_init * dy
                z = oz + t_init * dz
                s_new = self.g(x, y) + self.d - z
                item_t0 = (1 / (self.c * torch.clamp(1 + self.k, min=1e-8) ** 0.5)).abs() - (ox ** 2 + oy ** 2) ** 0.5
                # 通过逼近法先求解一个合适的t
                s_new[item_t0 <= 0] = 1
                item_back = (s_old * s_new) < 0
                t_init[item_back] = (t_init - rate)[item_back]
                rate[item_back] = rate[item_back] * 0.5
                s_old[~item_back] = s_new[~item_back]
            t_delta_init = t_init - t0

        self.z_con = t_delta_init * dz
        t1 = t0 + t_delta_init
        # time2 = time.time()
        with torch.no_grad():
            t, t_delta, valid = self.newtons_method_impl(
                maxt, t1, dx, dy, dz, ox, oy, oz, A, B, C
            )

        s_derivatives_dot_D = self.surface_and_derivatives_dot_D(
            t, dx, dy, dz, ox, oy, t_delta * dz, A, B, C
        )[1]
        t = t1 + t_delta
        residual = (self.g(ox + t * dx, oy + t * dy) + self.h(oz + t * dz) + self.d)

        # s_derivatives_dot_D[s_derivatives_dot_D.abs() < 1e-12] = 1e-12
        # s_derivatives_dot_D[s_derivatives_dot_D.abs() > 1e24] = 1e24
        # residual[residual.abs() < 1e-24] = 1e-24
        # residual[residual.abs() > 1e12] = 1e12
        s_derivatives_dot_D[s_derivatives_dot_D == 0] = 1e-12
        t = t - residual / s_derivatives_dot_D

        p = o + t * D
        # time3 = time.time()
        # print(time3-time2)
        # print(time2-time1)
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


# 都要是tensor
class Aspheric(Surfaces):
    """
    非球面: https://en.wikipedia.org/wiki/Aspheric_self.
    c：曲率，曲率半径的导数
    k：二次圆锥系数
    ai：非球面系数
    """

    def __init__(self, c, d, r, k=None, ai=None):
        device = c.device
        Surfaces.__init__(self, r, d)
        self.c = c
        if k is not None:
            self.k = k
        else:
            self.k = torch.zeros_like(self.c)
        if ai is not None:
            self.ai = ai
        else:
            self.ai = None

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


class Ray:
    """
    定义一个几何光线。
    - o 为光线起点
    - d 为归一化光线方向
    """

    def __init__(self, o, d, wavelength):
        self.o = o
        self.d = d
        # scalar-version
        self.wavelength = wavelength  # [nm]
        self.mint = 1e-5  # [mm]
        self.maxt = 1e5  # [mm]

    def __call__(self, t):
        return self.o + t * self.d


def nV_to_AB(n, V):
    def ivs(a): return 1. / a ** 2

    lambdas = [656.3, 589.3, 486.1]
    V[V == 0] = 1e-6
    B = (n - 1) / V / (ivs(lambdas[2]) - ivs(lambdas[0]))

    A = n - B * ivs(lambdas[1])
    return A, B


class Material_lab:
    def __init__(self, read_path, wave_all):
        data = pd.read_excel(read_path).values
        self.name = ['air']
        self.nd = [1.000]
        self.vd = [math.inf]
        self.wave = np.array(wave_all) / 1000
        self.formula = [np.ones_like(self.wave)]
        i = 0
        while i < len(data):
            if isinstance(data[i, 0], str) and data[i, 0][:4] == 'Name':
                j = 1
                while i + j < len(data):

                    if isinstance(data[i + j, 1], str) and (data[i + j, 1][:2].strip() == '标准' or data[i + j, 1][:2].strip() == '首选'):
                        min_lamda = data[i+j][13]
                        max_lamda = data[i+j][14]
                        if min_lamda > min(self.wave) or max_lamda < max(self.wave):
                            j += 1
                            continue
                        formula_now = data[i + j][22:]

                        if formula_now[0].strip() == 'Schott':
                            n_lamda = (formula_now[2] + formula_now[4] * (self.wave**2) + formula_now[6] * (self.wave**(-2)) + formula_now[8] * (self.wave**(-4)) + formula_now[10] * (self.wave**(-6)) + formula_now[12] * (self.wave**(-8))) ** 0.5
                        elif formula_now[0].strip() == 'Conrady':
                            n_lamda = formula_now[2] + formula_now[4] / self.wave + formula_now[6] / (self.wave ** 3.5)
                        elif formula_now[0].strip() == 'Sellmeier1':
                            n_lamda = (1 + (formula_now[2] * (self.wave ** 2) / (self.wave ** 2 - formula_now[4])) + (formula_now[6] * (self.wave ** 2) / (self.wave ** 2 - formula_now[8])) + (formula_now[10] * (self.wave ** 2) / (self.wave ** 2 - formula_now[12]))) ** 0.5
                        else:
                            j += 1
                            continue

                        self.name.append(data[i+j][0].strip())
                        self.nd.append(data[i+j][2])
                        self.vd.append(data[i+j][3])
                        self.formula.append(n_lamda)
                        j += 1
                    else:
                        j += 1
                        continue
                break
            i += 1
        self.formula = np.array(self.formula)


class Material:
    """
    计算折射率的光学材料
    n(波长) = A + B / 波长^2
    其中两个常数A和 B可以根据nD（589.3nm处的折射率）和 V（阿贝数）计算。
    """

    def __init__(self, name=None, data=None, pop=None, MATERIAL_TABLE=None, use_real_glass=None):
        self.A = None
        self.B = None
        self.MATERIAL_TABLE = MATERIAL_TABLE
        self.name_idx = None
        self.n_real = None
        # 这张表是编码的。
        if name is not None and pop is not None:
            device = pop.device
            size_pop = pop.size(1)
            self.n = torch.ones(size_pop).to(device) * self.MATERIAL_TABLE.nd[self.MATERIAL_TABLE.name.index(name)]
            self.V = torch.ones(size_pop).to(device) * self.MATERIAL_TABLE.vd[self.MATERIAL_TABLE.name.index(name)]
        if data is not None:
            self.n = data[0]
            self.V = data[1]
            if use_real_glass:
                device = self.n.device
                gt_nd = torch.Tensor(self.MATERIAL_TABLE.nd).to(device)
                gt_vd = torch.Tensor(self.MATERIAL_TABLE.vd).to(device)
                idx = ((self.n.unsqueeze(1) - gt_nd.unsqueeze(0)).square() + (
                        self.V.unsqueeze(1) - gt_vd.unsqueeze(0)).square() * 0.01).min(dim=1)[1]
                self.name_idx = idx
                self.n_real = torch.Tensor(self.MATERIAL_TABLE.formula).to(device)[idx]

    def ior(self, wavelength):
        """由给定的波长计算折射率 (in [nm])"""
        if self.n_real is not None:
            idx = np.argmin(abs(wavelength/1000-self.MATERIAL_TABLE.wave))
            n_final = self.n_real[:, idx]
        else:
            self.A, self.B = nV_to_AB(self.n, self.V)
            n_final = self.A + self.B / wavelength ** 2
        return n_final

    def to_string(self):
        return f'{self.A} + {self.B}/lambda^2'
