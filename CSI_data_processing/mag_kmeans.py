# - encoding=utf-8 -
# Kmeans: 正确率100%

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from csi_tool import read_csi

def main():
	# dataSize = 600 * (3 * 114)
	vMags, vPhases = read_csi('csiraw_phase_no1', 'csiraw_amp_no1')
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
	
	# Phase 中心化
	# for i in range(len(vPhase)):
	# 	vPhase[i] = vPhase[i] - vPhase[i][0]
	# 	vPhases[i] = vPhases[i] - vPhases[i][0]
	vMag = vMag[:100,]
	data_two_spots = np.concatenate((vMags, vMag), axis=0)

	# kmeans
	kmeans = KMeans(n_clusters=2, random_state=0).fit(data_two_spots)

	plt.figure(figsize=(12,8))
	p1 = plt.subplot()
	for i in range(len(data_two_spots)):
		if kmeans.labels_[i]:
			p1.plot(data_two_spots[i], color='red')
		else:
			p1.plot(data_two_spots[i], color='blue')
	p1.set_title('Kmeans of magnitudes in 2 different spots')
	p1.set_xlabel("subcarrier")
	p1.set_ylabel("magnitude")
	plt.savefig('figure/kmeans-mag-in-2-spots.png')
	plt.show()

if __name__ == '__main__':
	main()
