# - encoding=utf-8 -
# Locally Linear Embeddings Comparison
# 看起来一般的线性和非线性降维结果都不好

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import manifold, datasets
from csi_tool import read_csi

def main():
	# dataSize = 600 * (3 * 114)
	vMag, vPhase = read_csi('csiraw_phase', 'csiraw_amp')

	vMag = vMag.astype('float32') / 60.
	vMag = vMag.reshape((len(vMag), np.prod(vMag.shape[1:])))	# 600 * 342
	vPhase = vPhase.astype('float32') / np.pi
	vPhase = vPhase.reshape((len(vPhase), np.prod(vPhase.shape[1:])))

	# Locally Linear Embeddings
	X = vMag[:100,]
	X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12,
                                             n_components=5)
	plt.figure(figsize=(12,8))
	p1 = plt.subplot(211)
	p2 = plt.subplot(212)

	for i in range(100):
		p1.plot(X[i])
		p2.plot(X_r[i])

	p1.set_title('Original Data')
	p1.set_xlabel("subcarrier")
	p2.set_xlabel("reduced dim")
	p1.set_ylabel("mag")
	p2.set_ylabel("mag")

	plt.savefig('figure/feature-lle-mag.png')
	plt.show()

if __name__ == '__main__':
	main()
