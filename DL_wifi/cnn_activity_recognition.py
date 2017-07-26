# coding=utf-8
# Author=Jianfei

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# 超参数
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# 网络参数
data_size = 28
n_input = data_size * data_size
n_classes = 10
dropout = 0.8

# 占位符
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 卷积操作
def conv2d(name, l_input, w, b):
	return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='SAME'),b), name=name)

# 最大向下采样
def max_pool(name, l_input, k):
	return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME', name=name)

# 归一化操作
def norm(name, l_input, lsize=4):
	return tf.nn.lrn(l_input, lsize, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name=name)


def wifi_net(_X, _weights, _biases, _dropout):
	''' Model for WiFi based Activity Recognition'''
	_X = tf.reshape(_X, shape=[-1, data_size, data_size, 1])

	# Convolutional Layer #1
	conv1 = conv2d('conv1', _X, _weights['wc1'], _biases['bc1'])
	pool1 = max_pool('pool1', conv1, k=2)
	norm1 = norm('norm1', pool1, lsize=4)
	norm1 = tf.nn.dropout(norm1, _dropout)

	# Convolutional Layer #2
	conv2 = conv2d('conv2', norm1, _weights['wc2'], _biases['bc2'])
	pool2 = max_pool('pool2', conv2, k=2)
	norm2 = norm('norm2', pool2, lsize=4)
	norm2 = tf.nn.dropout(norm2, _dropout)

	# Convolutional Layer #3
	conv3 = conv2d('conv3', norm2, _weights['wc3'], _biases['bc3'])
	pool3 = max_pool('pool3', conv3, k=2)
	norm3 = norm('norm3', pool3, lsize=4)
	norm3 = tf.nn.dropout(norm3, _dropout)

	# Dense Layer
	# 特征图 ---> 向量
	dense1 = tf.reshape(norm3, [-1, _weights['wd1'].get_shape().as_list()[0]])
	dense1 = tf.nn.relu(tf.matmul(dense1, _weights['wd1']) + _biases['bd1'], name='fc1')
	dense2 = tf.nn.relu(tf.matmul(dense1, _weights['wd2']) + _biases['bd2'], name='fc2')

	# Output Layer
	out = tf.matmul(dense2, _weights['out']) + _biases['out']
	return out

# Save all weights
weights = {
	'wc1': tf.Variable(tf.random_normal([3, 3, 1, 64])),
	'wc2': tf.Variable(tf.random_normal([3, 3, 64, 128])),
	'wc3': tf.Variable(tf.random_normal([3, 3, 128, 256])),
	'wd1': tf.Variable(tf.random_normal([4*4*256, 1024])),
	'wd2': tf.Variable(tf.random_normal([1024, 1024])),
	'out': tf.Variable(tf.random_normal([1024, n_classes]))
}
biases = {
	'bc1': tf.Variable(tf.random_normal([64])),
	'bc2': tf.Variable(tf.random_normal([128])),
	'bc3': tf.Variable(tf.random_normal([256])),
	'bd1': tf.Variable(tf.random_normal([1024])),
	'bd2': tf.Variable(tf.random_normal([1024])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}

# 构建模型
pred = wifi_net(x, weights, biases, keep_prob)

# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 初始化所有的共享变量
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# 开启一个训练
with tf.Session() as sess:
	sess.run(init)
	step = 1
	# Keep training until reach max iterations
	while step * batch_size < training_iters:
		batch_xs, batch_ys = mnist.train.next_batch(batch_size)
		# 获取批数据
		sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
		if step % display_step == 0:
			# 计算精度
			acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			# 计算损失值
			loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
			print "Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc)
		step += 1
	print "Optimization Finished!"
	# 计算测试精度
	print "Testing Accuracy:", sess.run(accuracy, feed_dict={x: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.})
	# 保存模型
	save_path="model.ckpt"
	saver.save(sess, save_path)
	print "Model Store"
