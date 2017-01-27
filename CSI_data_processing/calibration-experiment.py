# - encoding=utf-8 -
# 对比每个点的差异，并进行简单的分类预测。

import scipy.io as sio
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
from sklearn.cluster import KMeans
from csi_tool import read_csi

