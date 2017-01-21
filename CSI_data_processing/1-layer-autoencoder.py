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
magTrain = vMag[0:400, :]
magTest = vMag[400:600, :]

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
	nb_epoch=50,
	batch_size=64,
	shuffle=True,
	validation_data=(magTest, magTest))

encoded_mag = encoder.predict(magTest)
decoded_mag = decoder.predict(encoded_mag)

plt.figure(figsize=(12,8))
p1 = plt.subplot(211)
p2 = plt.subplot(212)

for i in range(100, 200):
	p1.plot(decoded_mag[i-100])
	p2.plot(decoded_mag[i])

p1.set_title('Comparison of decoded results between static and moving')
p1.set_xlabel("subcarrier")
p2.set_xlabel("subcarrier")
p1.set_ylabel("magnitude")
p2.set_ylabel("magnitude")

# Result
# fig, ax = plt.subplots()
# for i in range(100):
# 	ax.plot(magTest[i], 'b', label='Original')
# 	ax.plot(decoded_mag[i], 'r', label='Autoencoder')
#legend = ax.legend(loc='upper right', shadow=True)

# plt.show()
plt.savefig('figure/results-of-decoded-in-static-and-moving.png')
plt.show()