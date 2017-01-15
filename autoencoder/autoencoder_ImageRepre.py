# *-- coding=utf-8 --*
# Purpose: Construct an autoencoder to represent image
# Dataset: MNIST

from keras.layers import Input, Dense
from keras.models import Model

########### Model construction ############

encoding_dim = 32 # 32-dim feature

# input placeholder
input_img = Input(shape=(784,))
# encoded representation
encoded = Dense(encoding_dim, activation='relu')(input_img)
# lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)

# this model maps an input to its encoded representation
encoder = Model(input=input_img, output=encoded)

# this model maps an encoded representation to reconstruction
encoded_input = Input(shape=(encoding_dim,))
decoded_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoded_layer(encoded_input))

########### Train ############

# configure our model to use a per-pixel binary crossentropy loss, and the Adadelta optimizer
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()

# normalize all values to (0, 1) and flatten 28*28 to 784-d vector
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# train the autoencoder
autoencoder.fit(x_train, x_train,
				nb_epoch=50,
				batch_size=256,
				shuffle=True,
				validation_data=(x_test, x_test))

# test some data
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

########### Visualization ###########
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
	# display original
	ax = plt.subplot(2, n, i + 1)
	plt.imshow(x_test[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# display reconstruction
	ax = plt.subplot(2, n, i + 1 + n)
	plt.imshow(decoded_imgs[i].reshape(28, 28))
	plt.gray()
	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

plt.show()
