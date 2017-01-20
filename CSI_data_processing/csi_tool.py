# - encoding=utf-8 -
# 采集数据后，read2ma.m将数据存为mat，本程序用于处理csi数据

import scipy.io as sio
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

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

# Normalize the magnitude to (30, 50) and flatten to vector
vMag = vMag.astype('float32') / 50.
vMag = vMag.reshape((len(vMag), np.prod(vMag.shape[1:])))

# 500 for training and 100 for testing
magTrain = vMag[0:500, :]
magTest = vMag[500:600, :]

# plt.plot(vPhase[0], 'b')
# plt.show()

# autoencoder 提取特征
encoding_dim = 32 # 32-dim feature
input_img = Input(shape=(342,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(342, activation='sigmoid')(encoded)
autoencoder = Model(input=input_img, output=decoded)
# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoded_layer(encoded_input))

########### Train ############

# configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train the autoencoder
autoencoder.fit(magTrain, magTrain,
	nb_epoch=100,
	batch_size=256,
	shuffle=True,
	validation_data=(magTest, magTest))

encoded_mag = encoder.predict(magTest)
decoded_mag = decoder.predict(encoded_mag)

plt.plot(magTest[0], 'b')
plt.plot(decoded_mag[0], 'r')
plt.show()