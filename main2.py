import numpy as np
from scipy.optimize import minimize

def OptimalControlbyLSSVM(x1, alpha1, lambda_value, kernel, f, J_N):
    pass

def funcional(alpha, x, lambda_value, kernel, f, J_N, vf,N):
    return (x[2]-vf)**2 + lambda_value*(sum(alpha[i]**2 for i in range(N)))

def kernel(sigma,x , l, i):
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][1]-x[i][1])**2 + (x[l][2]-x[i][2])**2))

def f(x, u, T, N, alpha):
    z1 = x[1] + (T/N)*(x[2])
    z2 = x[2] + (T/N)*(-alpha*x[1] + x[2] + u)

def J_N():
    pass
