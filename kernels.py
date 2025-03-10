import numpy as np


class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        N = X.shape[0]
        M = Y.shape[0]
        if X.ndim == 1:
            X_repeat = X[:,np.newaxis]
            Y_repeat = Y[:,np.newaxis]
        else: 
            X_repeat = np.repeat(X[:,np.newaxis,:], M, axis=1)
            Y_repeat = np.repeat(Y[np.newaxis,:,:], N, axis=0)
        diff = X_repeat - Y_repeat
        diff = np.squeeze(diff)
        return np.exp(-np.sum(diff**2, axis=-1)/(2*self.sigma**2)) ## Matrix of shape NxM
    

class Linear:
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        gram = np.dot(X, Y.T)
        return gram ## Matrix of shape NxM

class Polynomial:
    def __init__(self, degree=2, r=0):
        self.degree = degree
        self.r = r
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        gram = (np.dot(X, Y.T) + self.r)**self.degree
        return gram ## Matrix of shape NxM

class Sigmoid:
    def __init__(self, sigma=1., r=0):
        self.sigma = sigma
        self.r = r
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        gram = np.tanh(self.sigma*np.dot(X, Y.T) + self.r)
        return gram

