import numpy as np
from scipy.optimize import minimize

#Constantes
T = 5
N = 100
vf = 4
lambda_value = 0.01
sigma = 1
gamma = 1

# Variables iniciales
iniciales = [3, 2, 1] # x0, v0, alpha0
vars0 = np.zeros((N, 3))
vars0[0] = iniciales

# Transformar las 3 columnas en un vector
vars0 = vars0.T.flatten()

# Control
u = np.zeros(N)

# Parámetros
parametros = [T, N, vf, lambda_value, sigma, gamma]

# Definición de la función objetivo
def obj_funct(vars, parametros):
    _, N, vf, lambda_value, _, _ = parametros
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]
    return (x[N-1][1]-vf)**2 + lambda_value*np.sum(alpha**2)

# Definición del kernel
def kernel(x, l, i, parametros):
    _, _, _, _, sigma, _ = parametros
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][0]-x[i][0])**2 + (x[l][1]-x[i][1])**2))

# Restricciones sobre x
def f_rest(vars, u, i, parametros):
    T, N, _, _, _ , gamma= parametros
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]
    return [ x[i+1][0] -  x[i][0] + (T/N)*(x[i][1]) , x[i+1][1] - x[i][1] + (T/N)*(-gamma*x[i][0] + x[i][1] + u[i]) ]

# Restricciones sobre u
def u_rest(vars, u, i, kernel, parametros):
    _, N, _, _, _, _ = parametros
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]
    return u[i] - sum(alpha[l]*kernel(x, l, i, parametros) for l in range(N))

# Definición de la función de optimización
def OptimalControlbyLSSVM(vars0, obj_funct, f_rest, u_rest, kernel, parametros):
    constraints = []

    # Restricciones sobre x
    for i in range(N-1):
        constraints.append({'type': 'eq', 'fun': lambda vars: f_rest(vars, u, i, parametros)})
    
    # Restricciones sobre u
    for i in range(N-1):
        constraints.append({'type': 'eq', 'fun': lambda vars: u_rest(vars, u, i, kernel, parametros)})

    
    sol = minimize(lambda vars: obj_funct(vars, parametros), vars0, method='SLSQP', constraints=constraints)
    return sol

# Ejecución de la optimización
result = OptimalControlbyLSSVM(vars0, obj_funct, f_rest, u_rest, kernel, parametros)
print(result)
import matplotlib.pyplot as plt

# Extraer las soluciones
x_sol = result.x[:2*N].reshape(2, N).T
alpha_sol = result.x[2*N:]

# Graficar las soluciones
plt.figure(figsize=(12, 6))

# Posición x
plt.subplot(2, 1, 1)
plt.plot(range(N), x_sol[:, 0], label='Posición x')
plt.xlabel('Tiempo')
plt.ylabel('Posición x')
plt.legend()
plt.grid()

# Velocidad v
plt.subplot(2, 1, 2)
plt.plot(range(N), x_sol[:, 1], label='Velocidad v', color='orange')
plt.xlabel('Tiempo')
plt.ylabel('Velocidad v')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()