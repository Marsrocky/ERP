# - encoding=utf-8 -
# 采集数据后，read2ma.m将数据存为mat，本程序用于处理csi数据

import scipy.io as sio
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

def read_csi(matPhase = 'csiraw_phase', matMag = 'csiraw_amp'):
	# 读入幅度和相位信息
	rawPhase = sio.loadmat(matPhase + '.mat')[matPhase][0]
	rawMag = sio.loadmat(matMag + '.mat')[matMag][0]

	# rawData: 3 * (600 * 114) --> vPhase: 600 * (3 * 114)
	vPhase = np.empty((600, 3, 114))
	vMag = np.empty((600, 3, 114))
	for i in range(600):
		for j in range(3):
			for k in range(114):
				vPhase[i][j][k] = rawPhase[j][i][k]
				vMag[i][j][k] = rawMag[j][i][k]
	return vMag, vPhase

def main():
	# dataSize = 600 * (3 * 114)
	vMags, vPhases = read_csi('csiraw_phase_s1', 'csiraw_amp_s1')
	vMag, vPhase = read_csi('csiraw_phase', 'csiraw_amp')

	# Normalize the magnitude to (60, 50) and flatten to vector
	vMags = vMags.astype('float32') / 60.
	vMags = vMags.reshape((len(vMags), np.prod(vMags.shape[1:])))	# 600 * 342
	vPhases = vPhases.astype('float32') / np.pi
	vPhases = vPhases.reshape((len(vPhases), np.prod(vPhases.shape[1:])))

	vMag = vMag.astype('float32') / 60.
	vMag = vMag.reshape((len(vMag), np.prod(vMag.shape[1:])))	# 600 * 342
	vPhase = vPhase.astype('float32') / np.pi
	vPhase = vPhase.reshape((len(vPhase), np.prod(vPhase.shape[1:])))
	
	# 中心化
	for i in range(len(vPhase)):
		vPhase[i] = vPhase[i] - vPhase[i][0]
		vPhases[i] = vPhases[i] - vPhases[i][0]
	
	#################################################################

	# Comparison of magnitude between static and moving in 1min
	plt.figure(figsize=(12,8))
	p1 = plt.subplot(211)
	p2 = plt.subplot(212)

	for i in range(100):
		p1.plot(vPhase[i])
		p2.plot(vPhases[i])

	p1.set_title('Comparison of phase in 2 different spots')
	p1.set_xlabel("subcarrier")
	p2.set_xlabel("subcarrier")
	p1.set_ylabel("phase")
	p2.set_ylabel("phase")

	plt.savefig('figure/phase-in-2-spots.png')
	plt.show()

	# # Comparison of phase between static and moving in 1min
	# plt.figure(figsize=(12,8))
	# p1 = plt.subplot(211)
	# p2 = plt.subplot(212)

	# for i in range(100, 200):
	# 	p1.plot(vPhase[i-100])
	# 	p2.plot(vPhase[i])

	# p1.set_title('Comparison of magnitude between static and moving')
	# p1.set_xlabel("subcarrier")
	# p2.set_xlabel("subcarrier")
	# p1.set_ylabel("phase")
	# p2.set_ylabel("phase")

	# plt.savefig('figure/phase-between-static-and-moving.png')
	# plt.show()

if __name__ == '__main__':
	main()