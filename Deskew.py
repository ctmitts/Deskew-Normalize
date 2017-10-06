## Added some comments

import numpy as np
from scipy.stats import boxcox
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

mM = MinMaxScaler(feature_range=( .5, 1.5))


class Deskew(BaseEstimator, TransformerMixin):
    #mM = MinMaxScaler(feature_range=( .5, 1.5))
    def __init__ (self,alpha=1):
        self.alpha = alpha
        self.lambdas = []
    def _reset(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        #mM = MinMaxScaler(feature_range=( .5, 1.5))
        X_minMax = mM.fit_transform(X) 
        boxed = list()
        for col in X_minMax.T:
            boxcoxed, lam = boxcox( col)
            self.lambdas.append(lam)
            boxed.append( boxcoxed)
        return np.array( boxed).T
    def inverse_transform(self, X, y=None):
        cols = [] 
        for col, lam in zip(X.T, self.lambdas):
            if lam != 0:
                reskewed_col = (lam*col + 1)**(1/lam)
            else:
                reskewed_col = np.exp(col)
            cols.append( reskewed_col)    
        un_boxed = np.array( cols).T
        original = mM.inverse_transform( un_boxed )
        return original
    def score(self,X,y):
        pass
    
   # def fit_transform(self, X,y):
        
    
    
    
    
    