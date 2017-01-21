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

# dataSize = 600 * (3 * 114)
vMag, vPhase = read_csi()

# Normalize the magnitude to (60, 50) and flatten to vector
vMag = vMag.astype('float32') / 60.
vMag = vMag.reshape((len(vMag), np.prod(vMag.shape[1:])))	# 600 * 342
vPhase = vPhase.astype('float32') / np.pi
vPhase = vPhase.reshape((len(vPhase), np.prod(vPhase.shape[1:])))

vPhase = vPhase[:, :114]
plt.figure(figsize=(12,8))
p = plt.subplot()
for i in range(4, 10):
	plt.plot(vPhase[i])

p.set_title('phase')
p.set_xlabel("subcarrier")
p.set_ylabel("phase")
plt.show()

# plt.savefig('figure/magnitude-between-static-and-moving.png')

#################################################################

# Comparison of magnitude between static and moving in 1min
# plt.figure(figsize=(12,8))
# p1 = plt.subplot(211)
# p2 = plt.subplot(212)

# for i in range(100, 200):
# 	p1.plot(vMag[i-100])
# 	p2.plot(vMag[i])

# p1.set_title('Comparison of magnitude between static and moving')
# p1.set_xlabel("subcarrier")
# p2.set_xlabel("subcarrier")
# p1.set_ylabel("magnitude")
# p2.set_ylabel("magnitude")

# plt.savefig('figure/magnitude-between-static-and-moving.png')
# plt.show()

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