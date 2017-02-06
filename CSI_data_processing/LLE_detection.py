# - encoding=utf-8 -
# 对比每个点的差异，并进行简单的分类预测。

from time import clock
import matplotlib
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn import manifold, datasets, neighbors
from numpy import linalg as LA
from csi_tool import read_csi

sample_num = 200

def graph_feature(origin_mat, feature_mat):
	colors = ['red', 'blue', 'green', 'purple', 'cyan', 'magenta']
	fig = plt.figure(figsize=(12, 6))
	p1 = plt.subplot(211)
	p2 = plt.subplot(212)
	p1.set_title('Original Data')
	p2.set_title('Projected Data')
	for i in range(sample_num):
		p1.plot(origin_mat[i], color=colors[i/100])
		p2.plot(feature_mat[i], color=colors[i/100])
	plt.savefig('figure/LLE-static.png')
	plt.show()

def main():
	vMag, vPhase = read_csi('csiraw_phase', 'csiraw_amp')
	vMag = vMag.astype('float32') / 60.
	vMag = vMag.reshape((len(vMag), np.prod(vMag.shape[1:])))	# 600 * 342
	vPhase = vPhase.astype('float32') / np.pi
	vPhase = vPhase.reshape((len(vPhase), np.prod(vPhase.shape[1:])))

	X1 = vMag[:200,:114]
	feature_mat, err = manifold.locally_linear_embedding(X1, n_neighbors=10,
                                             n_components=8)
	graph_feature(X1, feature_mat)

if __name__ == '__main__':
	main()