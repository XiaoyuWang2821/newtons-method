"""
Newton's Method

"""

import numpy as np


def _linesearch(x, deltax, obj, fgrad, alpha, beta, debug=0):
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
    t = 1.0

    # other initialization
    fx = obj(x)
    innerprod = fgrad.T.dot(deltax)

    # while step length is too big
    while (np.isfinite(obj(x + t*deltax)) & (obj(x + t*deltax) > (fx + alpha * t * innerprod))):

        # decrease step length
        t = beta * t

    if debug > 1:
        print('\n------')
        print('-- (line search) t: %e \t fprev: %5.4f \t fnew: %5.4f \t innerprod: %5.4f  --' % (t, fx, obj(x + t*deltax), innerprod))
        print('------')

    return x + t*deltax


def _update(xk, oracle, rho, alpha=0.01, beta=0.5, debug=0):
    """
    Computes the Newton update step, with backtracking line search

    Parameters
    ----------
    xk : array_like
        The parameter vector at the current iteration

    oracle : function
        A function that takes a vector of parameters and returns an objective (scalar), gradient (vector), and Hessian (matrix)

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

    # clip negative eigenvalues (for non-convex problems)
    eigvals, eigvecs = np.linalg.eigh(H)
    H_clipped = eigvecs.dot(np.diag(eigvals * (eigvals > 0)).dot(eigvecs.T))

    # compute the search direction
    deltax = -np.linalg.solve(H_clipped + rho * np.eye(xk.size), fgrad)

    # check for bad values
    if debug > 0:
        if np.any(np.isnan(deltax)):
            print('\n*** ERROR *** found NaNs in search direction')
            1/0

        if np.any(np.isinf(deltax)):
            print('\n*** ERROR *** found Infs in search direction')

    # line search
    obj = lambda x: oracle(x, compute_grads=False)[0]
    x_new = _linesearch(xk, deltax, obj, fgrad, alpha, beta)

    # the norm of the gradient
    gradnorm = np.sqrt(fgrad.T.dot(H_clipped.dot(fgrad)))

    # print info about new location
    if debug:
        print('\n------')
        print('-- (new iterate) norm: %5.4f \t percent finite: %2.2f \t gradnorm: %5.4f --' % (np.linalg.norm(x_new), 100*np.mean(np.isfinite(x_new)), gradnorm))
        print('------')

    # check for bad values
    if debug > 0:
        if np.any(np.isnan(x_new)):
            print('\n*** ERROR *** found NaNs in new iterate after line search')

        if np.any(np.isinf(x_new)):
            print('\n*** ERROR *** found Infs in new iterate after line search')

    return x_new, fval, gradnorm


def optimize(x0, oracle, rho=1e-3, maxiter=20, tol=1e-2, debug=0):
    """
    Optimize a function using Newton's method

    Parameters
    ----------
    x0 : array_like

    oracle : function

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
        xk, fval, gradnorm = _update(xprev, oracle, rho)

        # store
        fvals.append(fval)
        gradnorms.append(gradnorm)

        # update
        if debug == 1:
            print('\n[%i] %5.4f' % (k+1, fval))
        elif debug > 1:
            print('\n[%i] f = %5.4f\t||grad|| = %5.4f\tstep size = %5.4f' % (k+1, fval, gradnorm, np.linalg.norm(xk-xprev)))

        # check if tolerance is reached
        if gradnorm <= tol:
            print('Converged after %i iterations!' % k)
            break

        # update parameters
        xprev = xk

    return xk, fvals, gradnorms
