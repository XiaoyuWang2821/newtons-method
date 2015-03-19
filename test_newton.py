"""
test Newton's method implementation

"""

import numpy as np
from newton import optimize
from functools import partial

def logistic_regression(theta, X, y, compute_grads=True):

    u = X.dot(theta)
    eu = np.exp(-u)
    p = 1 / (1 + eu)

    fval = np.sum(np.log(1 + eu) + u * (1 - y))

    fgrad = None
    fhess = None
    if compute_grads:
        fgrad = X.T.dot(p - y)
        fhess = X.T.dot(np.diag(eu * p**2).dot(X))

    return fval, fgrad, fhess

def linsys(theta, A, b, compute_grads=True):

    H = A.T.dot(A)
    v = A.T.dot(b)

    fval = 0.5 * theta.T.dot(H).dot(theta) - theta.T.dot(v)

    fgrad = None
    fhess = None
    if compute_grads:
        fgrad = H.dot(theta) - v
        fhess = H

    return fval, fgrad, fhess

def test_logistic(n=100, m=5000):

    # generate problem instance
    theta_star = np.random.randn(n)
    X = 2*np.vstack((np.random.randn(m-1,n), np.ones((1,n))))
    p_star = 1 / (1 + np.exp(-X.dot(theta_star)))
    y = (np.random.rand(m) < p_star).astype(float)

    # build the objective
    obj = partial(logistic_regression, X=X, y=y)

    # minimize using Newton's method
    xk, fvals, gradnorms = optimize(0.1*np.random.randn(n), obj, stepsize=0.9, rho=0.1)

    return xk, theta_star, fvals, gradnorms

def test_linsys(n=500, m=2000):

    A = np.random.randn(m,n)
    theta_star = np.random.randn(n)
    b = A.dot(theta_star) + 0.1*np.random.randn(m)

    theta_lsq = np.linalg.lstsq(A, b)[0]

    obj = partial(linsys, A=A, b=b)
    xk, fvals, gradnorms = optimize(0.1*np.random.randn(n), obj)

    return xk, theta_lsq, theta_star, fvals, gradnorms

if __name__ == "__main__":

    #xk, theta_lsq, theta_star, fvals, gradnorms = test_linsys()
    xk, theta_star, fvals, gradnorms = test_logistic()
