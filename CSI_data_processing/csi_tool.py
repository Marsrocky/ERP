# - encoding=utf-8 -
# 采集数据后，read2ma.m将数据存为mat，本程序用于处理csi数据

import scipy.io as sio
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

def read_csi():
	# 读入幅度和相位信息
	matPhase = 'csiraw_phase.mat'
	matMag = 'csiraw_amp.mat'
	rawPhase = sio.loadmat(matPhase)['csiraw_phase'][0]
	rawMag = sio.loadmat(matMag)['csiraw_amp'][0]

	# rawData: 3 * (600 * 114) --> vPhase: 600 * (3 * 114)
	vPhase = np.empty((600, 3, 114))
	vMag = np.empty((600, 3, 114))
	for i in range(600):
		for j in range(3):
			for k in range(114):
				vPhase[i][j][k] = rawPhase[j][i][k]
				vMag[i][j][k] = rawMag[j][i][k]
	return vMag, vPhase
