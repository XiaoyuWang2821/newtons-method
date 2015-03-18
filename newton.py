"""
Newton's Method
"""

import numpy as np

def update(xk, oracle, stepsize, rho):
    """
    Computes the Newton update step

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
    fval, df, H = oracle(xk)

    # compute the new update
    x_new =  xk - np.linalg.solve(H + rho * np.eye(xk.size), stepsize * df)

    # the norm of the gradient
    gradnorm = np.sqrt(df.T.dot(H.dot(df)))

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
