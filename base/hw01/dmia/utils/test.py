from  emath import _logistic_func
from scipy.sparse import *
import numpy as np

y_proba = np.empty((2, 2))
#print emath._logistic_func(csr_matrix(np.array(([10,20],[30,40]))),[1,2])
print (np.dot([1,2], np.array(([10,20],[30,40])))[:,np.newaxis])

print np.array(([10,20],[30,40]))
y = _logistic_func(csr_matrix(np.array(([10,20],[30,40]))),[1,2])
y2 = np.ones_like(y) - y
print np.hstack((y,y2))
#print y_proba[0]