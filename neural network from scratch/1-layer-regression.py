import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential
from keras.layers import Dense


n_samples = 1000
n_outliers = 50


X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1,
                                      n_informative=1, noise=10,
                                      coef=True, random_state=0)

X_train, Y_train = X[:800], y[:800]     # first 800 data points
X_test, Y_test = X[800:], y[800:]       # last 200 data points

model = Sequential()
model.add(Dense(output_dim=1, input_dim=1))

model.compile(loss='mse', optimizer='sgd')

print 'Training -----------'
for step in range(301):
	cost = model.train_on_batch(X_train, Y_train)
	if step % 100 == 0:
		print 'train cost: ', cost

# testing
print('\nTesting ------------')
cost = model.evaluate(X_test, Y_test, batch_size=40)
print('test cost:', cost)
W, b = model.layers[0].get_weights()
print('Weights=', W, '\nbiases=', b)

Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test)
plt.plot(X_test, Y_pred)
plt.show()