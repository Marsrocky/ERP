# - encoding=utf-8 -

import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np

from csi_tool import read_csi


# dataSize = 600 * (3 * 114)
vMag, vPhase = read_csi()

# Normalize the magnitude to (30, 60) and flatten to vector
vMag = vMag.astype('float32') / 60.
vMag = vMag.reshape((len(vMag), np.prod(vMag.shape[1:])))	# 600 * 342

# 500 for training and 100 for testing
magTrain = vMag[0:500, :]
magTest = vMag[500:600, :]

# autoencoder 提取特征
ncoding_dim = 16 # 32-dim feature

# model
input_img = Input(shape=(342,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)
encoded = Dense(16, activation='relu')(encoded)

decoded = Dense(32, activation='relu')(encoded)
decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(342, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

########### Train ############

# configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

# train the autoencoder
autoencoder.fit(magTrain, magTrain,
	nb_epoch=50,
	batch_size=256,
	shuffle=True,
	validation_data=(magTest, magTest))

decoded_mag = autoencoder.predict(magTest)

# Result
fig, ax = plt.subplots()
for i in range(100):
	ax.plot(magTest[i], 'b', label='Original')
	ax.plot(decoded_mag[i], 'r', label='Autoencoder')

# legend = ax.legend(loc='upper right', shadow=True)

plt.savefig('deep-layers-autoencoder.png')
plt.show()
