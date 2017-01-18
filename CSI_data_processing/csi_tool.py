# - encoding=utf-8 -
# 采集数据后，read2ma.m将数据存为mat，本程序用于处理csi数据

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 读入幅度和相位信息

matfn = 'magangle.mat'
data = sio.loadmat(matfn)

phaseData = data['phase']
magData = data['magnitude']