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




import numpy as np
from scipy.optimize import minimize

#Constantes
T = 5
N = 100
vf = 4
lambda_value = 0.01
sigma = 1

# Variables iniciales
x0 = np.array([0, 0])
alpha0 = np.array([1])
vars0 = np.hstack((x0, alpha0))

# Variables
x = np.zeros((N, 2))
alpha = np.zeros(N)

# Control
u = np.zeros(N)

# Parámetros
parametros = [T, N, vf, lambda_value, sigma]

def funcional(alpha, x, parametros):
    _, N, vf, lambda_value, _ = parametros
    return (x[N-1][1]-vf)**2 + lambda_value*np.sum(alpha**2)

def kernel(x, l, i, sigma):
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][0]-x[i][0])**2 + (x[l][1]-x[i][1])**2))

def restricciones(x, alpha, u, kernel, parametros):
    T, N, vf, lambda_value, sigma = parametros
    for i in range(N):
        x[i+1][0] = x[i][0] + (T/N)*(x[i][1])
        x[i+1][1] = x[i][1] + (T/N)*(-alpha*x[i][0] + x[i][1] + u[i])
        u[i] = np.sum(alpha[l]*kernel(sigma, x, l, i) for l in range(N))
    return np.array([x[i+1][0], x[i+1][1], u[i]])



def OptimalControlbyLSSVM(vars0, , kernel, restricciones, funcional, parametros):
    T, N, vf, lambda_value, sigma = parametros



    # Resolver el problema de optimización
    sol = minimize(objective, vars0, constraints=cons, method='SLSQP')

    return sol

# Llamar a la función de optimización
# sol = OptimalControlbyLSSVM(x0, alpha0, lambda_value, kernel, f, funcional, p)
# print(sol)




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





