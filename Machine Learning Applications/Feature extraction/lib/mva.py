# -*- coding: utf-8 -*-
"""
Author: Vanessa Gómez Verdejo (http://vanessa.webs.tsc.uc3m.es/)
Updated: 27/02/2017 (working with sklearn 0.18.1)
"""

import numpy as np


class mva:

    """
    MVA framework to solve PCA, CCA and OPLS approaches following the methods described in:
    Sergio Muñoz-Romero, Jerónimo Arenas-García, Vanessa Gómez-Verdejo,
    ``Sparse and kernel OPLS feature extraction based on eigenvalue problem solving''. Pattern Recognition 48(5): 1797-1811 (2015)

    Parameters
    ----------
    __algorithm : 'PCA', 'CCA' or 'OPLS'
    __n_components : int, (default 2).
        number of components to extract.
    __reg : int, (default 1e-3).
        regularization parameter to be used in covariance or kernel matrix inversion.
    Attributes
    ----------
    x_weights_ : array, [p, n_components]
        X block weights vectors.
    Example
    --------
    >>> from lib.mva import mva
    >>> CCA = mva('CCA', 10)
    >>> CCA.fit(X, Y, reg= 1e-3)
    >>> X_t = cca.transform(X)
    """
    __algorithm = None
    __n_components = 2
    __reg = 1e-3
    __x_weights = None

    def __init__(self, algorithm, n_components):

        # Check algorithm PCA, CCA or OPLS con try except
        self.__algorithm = algorithm
        self.__n_components = n_components

    def fit(self, X, Y, reg=None):
        if reg:
            self.__reg = reg

        N, dim = X.shape
        N, n_classes = Y.shape

        if self.__algorithm == 'PCA':
            Y = X
            R = np.eye(Y.shape[1])

        elif self.__algorithm == 'CCA':
            R = np.diag(np.power((1./N)*(np.diag(np.dot(Y.T, Y))), -.5))
            if self.__n_components > n_classes-1:
                print 'Maximum number of new projections fixed to ' + str(n_classes-1)
                self.__n_components = n_classes-1

        elif self.__algorithm == 'OPLS':
            R = np.eye(Y.shape[1])
            if self.__n_components > n_classes-1:
                print 'Maximum number of new projections fixed to ' + str(n_classes-1)
                self.__n_components = n_classes-1

        Cyx = (1./N) * np.dot(Y.T, X)

        if dim < N:
            Cxx = (1./N)*(np.dot(X.T, X) + self.__reg * np.eye(dim))
            pinvXY = np.linalg.lstsq(Cxx, Cyx.T)[0]
        else:
            Kxx = np.dot(X, X.T) + self.__reg * np.eye(N)
            pinvXY = np.dot(X.T, np.linalg.solve(Kxx, Y))

        A = np.dot(np.dot(R, np.dot(Cyx, pinvXY)), R)
        W, D, V = np.linalg.svd(A)

        pos = np.argsort(D)[::-1]
        W = W[:, pos[:self.__n_components]]
        #val = D[pos[:self.__n_components]]
        W = np.dot(R, W)
        self.__x_weights = np.dot(pinvXY, W)

        # Checking whitenning of the data
        X_t = self.transform(X)
        AA = np.dot(X_t.T, X_t) / N      
        AAideal = np.diag(D[:self.__n_components])
        
        if np.mean(np.abs(AAideal -AA)) > 1e-3:
            print 'Data are ill-conditioned, please increase the value of the regularization parameter'


    def transform(self, X):
        return np.dot(X, self.__x_weights)
