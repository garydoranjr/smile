"""
Wraps sklearn SVM classes to allow
for set kernels
"""
from numpy import matrix, vstack, hstack
import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from scipy.optimize import fmin_bfgs as fmin
import math

from progress import ProgressMonitor

MEM_LIMIT = 1024*1024*1024 # 1GB

class SetSVM(object):

    def __init__(self, estimator_class, set_kernel, **kwargs):
        self.set_kernel = _by_name(set_kernel)
        self.estimator = estimator_class(**kwargs)

    def fit(self, X, y):
        X = map(np.asmatrix, X)
        self.fit_data = X
        gram_matrix = self.set_kernel(X, X)
        self.estimator.fit(gram_matrix, y)
        return self

    def predict(self, X):
        gram_matrix = self.set_kernel(X, self.fit_data)
        return self.estimator.predict(gram_matrix)

    def decision_function(self, X):
        gram_matrix = self.set_kernel(X, self.fit_data)
        return self.estimator.decision_function(gram_matrix)

def _by_name(full_name):
    parts = full_name.split('_')
    name = parts.pop(0)

    try:
        # See if second part is a number
        value = float(parts[0])
        parts.pop(0)
    except: pass

    if name == 'linear':
        kernel = linear
    elif name == 'quadratic':
        kernel = quadratic
    elif name == 'p':
        kernel = polynomial(int(value))
    elif name == 'rbf':
        kernel = rbf(value)
    else:
        raise ValueError('Unknown Kernel type %s' % name)

    try:
        # See if remaining part is a norm
        norm_name = parts.pop(0)
        if norm_name == 'fs':
            norm = featurespace_norm
        elif norm_name == 'av':
            norm = averaging_norm
        elif norm_name == 'md':
            norm = 'median'
        else:
            raise ValueError('Unknown norm %s' % norm_name)
    except IndexError:
        norm = no_norm

    kernel_function = set_kernel(kernel, norm)
    kernel_function.name = full_name
    return kernel_function

def averaging_norm(x, *args):
    return float(x.shape[0])

def featurespace_norm(x, k):
    return math.sqrt(np.sum(k(x, x)))

def no_norm(x, k):
    return 1.0

def _prog(plist):
    progress = ProgressMonitor(total=len(plist), print_interval=1,
                               msg='Constructing Kernel')
    for p in plist:
        yield p
        progress.increment()

def set_kernel(k, normalizer=no_norm):
    """
    Decorator that makes a normalized
    set kernel out of a standard kernel k
    """
    # Check special case
    # (kind of a hack; make it better eventually)
    if normalizer == 'median':
        return median_kernel(k)

    def K(X, Y):
        if type(X) == list:
            norm = lambda x: normalizer(x, k)
            xinst = sum(map(len, X))
            yinst = sum(map(len, Y))
            if xinst*yinst*8 >= MEM_LIMIT:
                x_norm = matrix(map(norm, X))
                y_norm = matrix(map(norm, Y))
                norms = x_norm.T*y_norm
                raw_kernel = np.array([[np.sum(k(x,y)) for y in Y]
                                       for x in _prog(X)])
            else:
                x_norm = matrix(map(norm, X))
                if id(X) == id(Y):
                    # Optimize for symmetric case
                    norms = x_norm.T*x_norm
                    if all(len(bag) == 1 for bag in X):
                        # Optimize for singleton bags
                        instX = vstack(X)
                        raw_kernel = k(instX, instX)
                    else:
                        # Only need to compute half of
                        # the matrix if it's symmetric
                        upper = matrix([i*[0] + [np.sum(k(x, y))
                                                 for y in Y[i:]]
                                        for i, x in enumerate(X, 1)])
                        diag = np.array([np.sum(k(x, x)) for x in X])
                        raw_kernel = upper + upper.T + spdiag(diag)
                else:
                    y_norm = matrix(map(norm, Y))
                    norms = x_norm.T*y_norm
                    raw_kernel = k(vstack(X), vstack(Y))
                    lensX = map(len, X)
                    lensY = map(len, Y)
                    if any(l != 1 for l in lensX):
                        raw_kernel = vstack([np.sum(raw_kernel[i:j, :], axis=0)
                                             for i, j in slices(lensX)])
                    if any(l != 1 for l in lensY):
                        raw_kernel = hstack([np.sum(raw_kernel[:, i:j], axis=1)
                                             for i, j in slices(lensY)])
            return np.divide(raw_kernel, norms)
        else:
            return k(X, Y)
    return K

def linear(x, y):
    """Linear kernel x'*y"""
    return np.dot(x, y.T)

def quadratic(x, y):
    """Quadratic kernel (1 + x'*y)^2"""
    return np.square(1e0 + np.dot(x,y.T))

def polynomial(p):
    """General polynomial kernel (1 + x'*y)^p"""
    def p_kernel(x, y):
        return np.power(1e0 + np.dot(x,y.T), p)
    return p_kernel

def rbf(gamma):
    """Radial Basis Function"""
    def rbf_kernel(x, y):
        return matrix(np.exp(-gamma*cdist(x, y, 'sqeuclidean')))
    return rbf_kernel

def slices(groups):
    """
    Generate slices to select
    groups of the given sizes
    within a list/matrix
    """
    i = 0
    for group in groups:
        yield i, i + group
        i += group

def spdiag(x):
    n = len(x)
    return sp.spdiags(x.flat, [0], n, n)

def median_weight(k, X):
    n = len(X)
    K = k(X, X)
    a0 = np.ones(n) / float(n)
    if n <= 2:
        return a0

    def distances(a):
        a = np.asarray(a)
        sq_dists = np.array([float(K[i, i] - 2 * np.dot(K[i, :], a) + np.dot(np.dot(a.T, K), a))
                             for i in range(n)])
        return np.sqrt(sq_dists)

    def f(a):
        return np.sum(distances(a))

    def grad(a):
        a = np.asarray(a)
        dists = distances(a)
        g = sum((np.dot(K, a) - K[i, :]) / dists[i]
                 for i in range(n) if dists[i] != 0)
        if type(g) == int:
            return np.zeros(a.shape)
        return np.array(list(g.flat))

    astar = fmin(f, a0, fprime=grad, disp=0)
    astar = np.asarray(astar)
    if np.any(np.isnan(astar)):
        return a0
    else:
        return astar

def median_kernel(k):
    """
    Makes a "median kernel" out of instance kernel k
    """
    def make_weights(X):
        n = len(X)
        prog = ProgressMonitor(total=n, print_interval=1,
                               msg='Constructing Kernel')
        ws = []
        for x in X:
            prog.increment()
            ws.append(median_weight(k, x))
        return ws

    def K(X, Y):
        X_meds = make_weights(X)
        if id(X) == id(Y):
            Y_meds = X_meds
        else:
            Y_meds = make_weights(Y)
        kernel = np.array([[float(np.dot(np.dot(a.T,k(x, y)),b))
                            for y, b in zip(Y, Y_meds)]
                           for x, a in zip(X, X_meds)])
        return kernel
    return K
