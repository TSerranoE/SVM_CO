import numpy as np
from scipy.optimize import minimize
import timeout_decorator

#Constantes
T = 10
N = 70
vf = 100
lambda_value = 0
sigma = 1
gamma = 1

# Variables iniciales
iniciales = [3, 100, 1] # x0, v0, alpha0
vars0 = np.zeros((N, 3))
vars0[0] = iniciales

# Transformar las 3 columnas en un vector
vars0 = vars0.T.flatten()


# Parámetros
parametros = [T, N, vf, lambda_value, sigma, gamma]

# Definición de la función objetivo
def obj_funct(vars, kernel, parametros):
    T, N, vf, lambda_value, _, gamma = parametros
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]

    x_gorro = np.zeros((N, 2))
    x_gorro[0] = iniciales[:2]
    # Restricciones sobre x
    for i in range(N-1):
        x_gorro[i+1][0] =  x_gorro[i][0] + (T/N)*(x_gorro[i][1]) 
        x_gorro[i+1][1] = x_gorro[i][1] + (T/N)*(-gamma*x_gorro[i][0] + x_gorro[i][1] + sum(alpha[l]*
                                                       kernel(x, x_gorro, l, i, parametros) for l in range(N))) 
    print("valor", (x_gorro[N-1][1]-vf)**2 + lambda_value*np.sum(alpha**2))
    return (x_gorro[N-1][1]-vf)**2 + lambda_value*np.sum(alpha**2)

# Definición del kernel
def kernel(x, x_gorro, l, i, parametros):
    _, _, _, _, sigma, _ = parametros
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][0]-x_gorro[i][0])**2 + (x[l][1]-x_gorro[i][1])**2))

# Restricciones sobre x
def f_rest(vars, u, i, x_gorro,parametros):
    T, N, _, _, _ , gamma= parametros
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]



# Restricciones sobre u
def u_rest(vars, u, i, kernel, parametros):
    _, N, _, _, _, _ = parametros
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]
    return u[i] - sum(alpha[l]*kernel(x, l, i, parametros) for l in range(N))

# Definición de la función de optimización
@timeout_decorator.timeout(300) 
def OptimalControlbyLSSVM(vars0, obj_funct, kernel, parametros):

    sol = minimize(lambda vars: obj_funct(vars, kernel,parametros), vars0, method='SLSQP')
    return sol

# Ejecución de la optimización

try:
    result = OptimalControlbyLSSVM(vars0, obj_funct, kernel, parametros)
    print(result)
except timeout_decorator.timeout_decorator.TimeoutError:
    print("La optimización se detuvo después de 300 segundos.")


import matplotlib.pyplot as plt

# Extraer las soluciones
x_sol = result.x[:2*N].reshape(2, N).T
alpha_sol = result.x[2*N:]

x_gorro = np.zeros((N, 2))
x_gorro[0] = iniciales[:2]
# Restricciones sobre x
for i in range(N-1):
    x_gorro[i+1][0] =  x_gorro[i][0] + (T/N)*(x_gorro[i][1]) 
    x_gorro[i+1][1] = x_gorro[i][1] + (T/N)*(-gamma*x_gorro[i][0] + x_gorro[i][1] + sum(alpha_sol[l]*
                                                    kernel(x_sol, x_gorro, l, i, parametros) for l in range(N))) 
print("valor", (x_gorro[N-1][1]-vf)**2 + lambda_value*np.sum(alpha_sol**2))
print("velocidad final", x_gorro[N-1][1])

# Graficar las soluciones
plt.figure(figsize=(12, 6))

# Posición x
plt.subplot(2, 1, 1)
plt.plot(range(N), x_gorro[:, 0], label='Posición x')
plt.xlabel('Tiempo')
plt.ylabel('Posición x')
plt.legend()
plt.grid()

# Velocidad v
plt.subplot(2, 1, 2)
plt.plot(range(N), x_gorro[:, 1], label='Velocidad v', color='orange')
plt.xlabel('Tiempo')
plt.ylabel('Velocidad v')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()