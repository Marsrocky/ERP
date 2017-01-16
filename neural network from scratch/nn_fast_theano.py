# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import theano
import theano.tensor as T
import time

# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func):
	# Set min and max values and give it some padding
	x_min, x_max = train_X[:, 0].min() - .5, train_X[:, 0].max() + .5
	y_min, y_max = train_X[:, 1].min() - .5, train_X[:, 1].max() + .5
	h = 0.01
	# Generate a grid of points with distance h between them
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# Predict the function value for the whole gid
	Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)
	# Plot the contour and training examples
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
	plt.scatter(train_X[:, 0], train_X[:, 1], c=train_y, cmap=plt.cm.Spectral)

####################################################################################
# Generate a dataset
np.random.seed(0)
train_X, train_y = sklearn.datasets.make_moons(200, noise=0.2)
train_X = train_X.astype(np.float32)
train_y = train_y.astype(np.int32)
plt.scatter(train_X[:,0], train_X[:,1], s=40, c=train_y, cmap=plt.cm.Spectral)

# Network size
num_examples = len(train_X)
nn_input_dim = 2	# input layer dimensionality
nn_output_dim = 2	# output layer dimensionality
nn_hdim = 100		# hidden layer dimensionality

# Gradient descent parameters
epsilon = 0.01	# learning rate for gd
reg_lambda = 0.01 # regularization strength

# Our data vectors
X = T.matrix('X')
y = T.lvector('y')
(X * 2).eval({X : [[1,1],[2,2]]})

# Shared variables with initial values
# 指明每个共享变量的维度以正确建图
W1 = theano.shared(np.random.randn(nn_input_dim, nn_hdim), name='W1')
b1 = theano.shared(np.zeros(nn_hdim), name='b1')
W2 = theano.shared(np.random.randn(nn_hdim, nn_output_dim), name='W2')
b2 = theano.shared(np.zeros(nn_output_dim), name='b2')

####################################################################################
# Forward propagation
# 网络的计算过程给出，从X到输出概率的映射，这里只定义不计算
z1 = X.dot(W1) + b1
a1 = T.tanh(z1)
z2 = a1.dot(W2) + b2
y_hat = T.nnet.softmax(z2)	# output probabilities

# The regularization term (optional)
loss_reg = 1./num_examples * reg_lambda/2 * (T.sum(T.sqr(W1)) + T.sum(T.sqr(W2))) 
# the loss function we want to optimize
loss = T.nnet.categorical_crossentropy(y_hat, y).mean() + loss_reg

# Returns a class prediction
prediction = T.argmax(y_hat, axis=1)

####################################################################################
# Create Theano functions
# function参数为函数输入、输出
forward_prop = theano.function([X], y_hat)
calculate_loss = theano.function([X, y], loss)
predict = theano.function([X], prediction, allow_input_downcast=True)

# 将建好的网络图保存 | 用text打印
theano.printing.pydotprint(forward_prop, var_with_name_simple=True, compact=True, outfile='img/nn-theano-forward_prop.png', format='png')
# theano.printing.debugprint(forward_prop)

# Calculate the derivatives
dW2 = T.grad(loss, W2)	# dL/dW2
db2 = T.grad(loss, b2)	# dL/db2
dW1 = T.grad(loss, W1)	# dL/dW1
db1 = T.grad(loss, b1)	# dL/db1

# 用X和y作为输入做backpropagation，更新一次参数
gradient_step = theano.function(
	[X, y],
	updates=((W2, W2 - epsilon * dW2),
			 (W1, W1 - epsilon * dW1),
			 (b2, b2 - epsilon * db2),
			 (b1, b1 - epsilon * db1)))
%timeit gradient_step()

# Build the model with learning parameters 
def build_model(num_passes = 20000):
	np.random.seed(0)
	# 除以了sqrt(input_num)使得随机产生的矩阵entry都在(0,1)中
	W1.set_value(np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim))
	b1.set_value(np.zeros(nn_hdim))
	W2.set_value(np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim))
	b2.set_value(np.zeros(nn_output_dim))

	# Gradient descent
	for i in xrange(0, num_passes):
		gradient_step(train_X, train_y)

		if i % 1000 == 0:
			print 'Loss after iteration:%i: %f' % (i, calculate_loss(train_X, train_y))

# Build a model with a 3-dimensional hidden layer
build_model()
plot_decision_boundary(lambda x:predict(x))
plt.title('Decision Boundary for hidden layer size 3')
plt.show()