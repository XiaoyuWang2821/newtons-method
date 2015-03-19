"""
Newton's Method

"""

import numpy as np

def linesearch(x, deltax, obj, fgrad, alpha, beta):
    """
    Line search

    Parameters
    ----------
    x : array_like

    deltax : array_like
        Search direction

    oracle : function

    fgrad : array_like

    alpha : float

    beta : float

    Returns
    -------
    x_next : array_like
        The new location

    """

    # initialize step size to 1
    t = 1

    # other initialization
    fx = obj(x)
    innerprod = fgrad.T.dot(deltax)

    # while step length is too big
    while obj(x + t*deltax) > (fx + alpha * t * innerprod):

        # decrease step length
        t = beta * t

    return x + t*deltax

def update(xk, oracle, stepsize, rho, alpha=0.01, beta=0.5):
    """
    Computes the Newton update step, with backtracking line search

    Parameters
    ----------
    xk : array_like
        The parameter vector at the current iteration

    oracle : function
        A function that takes a vector of parameters and returns an objective (scalar), gradient (vector), and Hessian (matrix)

    stepsize : float
        The step size parameter (positive, less than one)

    rho : float
        The damping parameter (positive). An identity matrix with rho on the diagonal is added to the Hessian during the Newton update

    Returns
    -------
    x_new : array_like
        The updated parameters after a Newton step

    fval : float
        The function value at this iteration

    gradnorm : float
        The norm of the gradient vector at this iteration, under the metric induced by the Hessian

    """

    # query the oracle for the objective, gradient, and hessian
    fval, fgrad, H = oracle(xk)

    # compute the search direction
    deltax = -np.linalg.solve(H + rho * np.eye(xk.size), stepsize * fgrad)

    # line search
    obj = lambda x: oracle(x, compute_grads=False)[0]
    x_new = linesearch(xk, deltax, obj, fgrad, alpha, beta)

    # the norm of the gradient
    gradnorm = np.sqrt(fgrad.T.dot(H.dot(fgrad)))

    return x_new, fval, gradnorm

def optimize(x0, oracle, stepsize=1, rho=1e-3, maxiter=20, tol=1e-6):
    """
    Optimize a function using Newton's method

    Parameters
    ----------
    x0 : array_like

    oracle : function

    stepsize : float, optional

    rho : float, optional

    maxiter : int, optional

    tol : float, optional

    """

    # initialize
    xprev = x0
    fvals = list()
    gradnorms = list()

    for k in range(maxiter):

        # compute the Newton update
        xk, fval, gradnorm = update(xprev, oracle, stepsize, rho)

        # check if tolerance is reached
        if np.linalg.norm(xk-xprev) <= tol:
            print('Converged after %i iterations!' % k)
            break

        # store
        fvals.append(fval)
        gradnorms.append(gradnorm)

        # update parameters
        xprev = xk

        # update
        print('[%i] %5.4f' % (k+1, fval))

    return xk, fvals, gradnorms
