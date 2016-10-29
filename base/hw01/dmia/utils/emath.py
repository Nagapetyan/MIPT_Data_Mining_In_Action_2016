import numpy as np
from scipy.sparse import csr_matrix


##########################################################
#	__inner_sigmoid and sigmoid no longer used			 #
##########################################################
def __inner_sigmoid(x):
		return 1./(1. + np.exp(-1.*x))

def sigmoid(X, w, out=None):
	data = csr_matrix(X.dot(w[:,np.newaxis]))

	if out == None:
		out = csr_matrix(data.shape)

	for i in xrange(data.shape[0]):
		for j in xrange(data.shape[1]):
			elem = np.array([__inner_sigmoid(data[i, j])])
			row = np.array([i])
			col = np.array([j])
			out += csr_matrix((elem, (row,col)), data.shape)
	return out


def log_logistic(X,w,y, out=None):
	if out == None:
		out = np.empty_like(y*X.dot(w))
	out = __inner_sigmoid(y*X.dot(w))
	return out