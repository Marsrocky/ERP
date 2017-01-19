# - encoding=utf-8 -
# 采集数据后，read2ma.m将数据存为mat，本程序用于处理csi数据

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# 读入幅度和相位信息
matPhase = 'csiraw_phase.mat'
matMag = 'csiraw_amp.mat'
rawPhase = sio.loadmat(matPhase)['csiraw_phase'][0]
rawMag = sio.loadmat(matMag)['csiraw_amp'][0]

# rawData: 3 * (600 * 114)
print rawPhase

# autoencoder 提取特征