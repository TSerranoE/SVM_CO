import numpy as np
from scipy.optimize import minimize


#Constantes
x0 = [0,0]
alpha0 = 1
alpha = 1
T = 5
N = 100
vf = 4
lambda_value = 0.01
sigma = 1

p = [alpha ,T, N, vf, lambda_value, sigma]

def funcional(alpha, x, lambda_value, vf, N):
    return (x[2]-vf)**2 + lambda_value*(sum(alpha[i]**2 for i in range(N)))

def kernel(sigma,x , l, i):
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][1]-x[i][1])**2 + (x[l][2]-x[i][2])**2))

def f(x, u, T, N, alpha):
    z1 = x[1] + (T/N)*(x[2])
    z2 = x[2] + (T/N)*(-alpha*x[1] + x[2] + u)
    return z1, z2

def u_restriccion(sigma, x, l):
    return sum(alpha[i]*kernel(sigma, x, l, i) for i in range(N))

def OptimalControlbyLSSVM(x0, alpha0, lambda_value, kernel, f, funcional, p):
    alpha ,T, N, vf, lambda_value, sigma = p

    sol = minimize(funcional, [x0,alpha0], args=(x0, alpha ,T, N, vf, lambda_value, sigma))

    return sol





