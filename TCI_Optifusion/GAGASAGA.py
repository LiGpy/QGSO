import numpy
import pickle
import time
import torch
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler
import argparse
import gaoptics as ga
from gaoptics.simulation_func import *
import pandas as pd

if __name__ == '__main__':
    torch.set_num_threads(8)
    """
    初始化镜头，超参数配置
    """
    len_pop = ga.LensPopulation()
    len_pop.basics.device = torch.device('cuda:' + str(0))

    len_pop.basics.fnum = 2.5
    len_pop.basics.acc_fit = 0.08
    len_pop.basics.acc_range = 0.01
    len_pop.basics.z_object = -1000000
    len_pop.basics.z_object_edof = [-1000000]
    len_pop.basics.img_r = 14.3
    len_pop.basics.fov = 20
    len_pop.basics.EFFL = (38, 42, 0.1)

    len_pop.basics.aper_index = 4
    len_pop.basics.structure = ['S0G', 'S0A', 'S0G', 'S0A', 'S0A', 'S0G', 'S0A']
    len_pop.basics.fix_vari = [[len_pop.basics.aper_index, 'c', 1e-10]]
    len_pop.basics.save_root = './exp/GAGASAGA.xlsx'
    # 物理约束
    len_pop.basics.CVVA = (-0.1, 0.1)
    len_pop.basics.CTVAG = (1.3, 8)
    len_pop.basics.CTVAA = (1.3, 10)
    len_pop.basics.INDX = (1.51, 1.76)
    len_pop.basics.MVAB = (27.5, 71.3)
    len_pop.basics.ETVAG = (1.0, 8, 1)
    len_pop.basics.ETVAA = (0.3, 8, 1)
    len_pop.basics.BFFL = (29.0, 40, 0.05)
    len_pop.basics.TTVA = (0.0, 50, 0.1)
    len_pop.basics.RIMH = (-0.01, 0.01, 1)
    len_pop.basics.DISG = (-0.01, 0.01, 1)

    len_pop.basics.max_r = 20
    len_pop.basics.demo_root = './init_structure/'
    len_pop.basics.init_read_root = None
    len_pop.basics.init_read_root = './exp/GAGASAGA.xlsx'

    len_pop.basics.material_root = 'CDGMtci.xlsx'

    # 仿真与优化配置
    len_pop.basics.fov_sample_num = 3
    len_pop.basics.enp_sample_num = 9
    len_pop.basics.max_pop = 4000
    len_pop.basics.sphere_gener_num = 60
    len_pop.basics.asphere_gener_num = 0
    len_pop.basics.wave_sample = {
        "R": ([656.0], [1]),
        "G": ([588.0], [1]),
        "B": ([486.0], [1]),
    }

    # 根据设定初始化参数
    len_pop.basics.use_aspheric = False
    len_pop.basics.use_doe = False

    ga_instance = ga.GA(len_pop=len_pop)
    time_start = time.time()
    ga_instance.run()
    time_end = time.time()
    print('time_all:{time}'.format(time=time_end - time_start))
