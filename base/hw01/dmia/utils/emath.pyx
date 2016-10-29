import numpy as np
from scipy.sparse import *

def __inner_sigmoid(x):
	if x > 0:
		return -np.log(1. + np.exp(-x))
	else:
		return x - np.log(1. + np.exp(x))


def __sigmoid(n_samples, n_features, X):
	out = np.zeros_like(X)
	for i in range(n_samples):
		for j in range(n_features):
			out[i][j] = __inner_sigmoid(X[i, j])
	return out

def _logistic_func(X, w):
	data = np.dot(w, X.toarray())[:,np.newaxis]
	return __sigmoid(data.shape[0], data.shape[1], data)