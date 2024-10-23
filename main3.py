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


# Definición de la función objetivo
def obj_funct(x, alpha):
    return (x[N]-vf)**2 + lambda_value*(sum(alpha[i]**2 for i in range(N)))


def f_rest(x, u, i):
    return [   x[i+1][0]   ,    x[i+1][1]    ] - [   x[i][0] + (T/N)*(x[i][1])    ,    x[i][1] + (T/N)*(-alpha*x[i][0] + x[i][1] + u[i])    ]



def u_rest(x, u, alpha, i):
    return u[i] - np.sum(alpha[l]*kernel(x, l, i, sigma) for l in range(N))




def OptimalControlbyLSSVM():
    
    constraints = []

    # Restricciones sobre x
    for i in range(N):
        constraints.append({'type': 'eq', 'fun': lambda x: f_rest(x, i)})
    
    # Restricciones sobre u
    for i in range(N):
        constraints.append({'type': 'eq', 'fun': lambda alpha: u_rest(alpha, i)})

    
    sol = minimize(obj_funct, (x0,alpha0), method='SLSQP', constraints=constraints)

    return sol

# Ejecución de la optimización
result = OptimalControlbyLSSVM()
print(result)
