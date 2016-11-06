#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator
from scipy.special import expit


class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, lr=0.1, max_depth=3):
        self.base_estimator = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators # +1 estimator with zero prediction
        self.feature_importances_ = None
        self.estimators_ = []
        self.coefs_ = []
    

    def loss_grad(self, original_y, pred_y):
        return -original_y * expit(-pred_y*original_y)

    def fit(self, X, original_y):
        self.estimators_ = []

        for i in range(self.n_estimators):
            grad = self.loss_grad(original_y, self._predict(X))

            estimator = deepcopy(self.base_estimator)
            estimator.fit(X, -grad)
            coef = -grad.dot(estimator.predict(X))/estimator.predict(X).dot(estimator.predict(X))
            self.coefs_.append(coef)
            self.estimators_.append(estimator)

        grad = self.loss_grad(original_y, self._predict(X))
        self.out_ = self._outliers(grad)
        self.feature_importances_ = self._calc_feature_imps()

        return self

    def _predict(self, X):
        y_pred = np.zeros(X.shape[0])
#        for estimator in self.estimators_:
#            y_pred+= self.lr*estimator.predict(X)
        for estimator, coef in zip(self.estimators_, self.coefs_):
            y_pred += coef*estimator.predict(X)
    
        return y_pred

    def predict(self, X):
        return np.sign(self._predict(X))

    def _outliers(self, grad):
        return np.hstack((np.argsort(grad)[:10], np.argsort(grad)[-10:]))

    def _calc_feature_imps(self):
        f_imps = np.zeros_like(self.estimators_[0].feature_importances_)
        
        for i in np.arange(len(self.estimators_)):
            f_imps += self.estimators_[i].feature_importances_*self.coefs_[i]#self.lr
        return f_imps/len(self.estimators_)
