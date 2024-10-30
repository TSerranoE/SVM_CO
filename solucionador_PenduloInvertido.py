import numpy as np
from Ej3_PenduloInvertido import funcion_objetivo, variable_inicial, kernel, plot
from scipy.optimize import minimize


#Constantes
T = 10
N = 50
lambda_value = 0.01
sigma = 1
h = 0.05
m = 0.1
m_t = 1.1
l = 0.5
g = 9.81
Q = np.eye(4)
R = 0.01
A = np.zeros((4,4))
B = np.zeros(4)

print(A)
print(B)


# Parámetros
parametros = [T, N, lambda_value, sigma, h, m, m_t, l, g, Q, R, A , B]

# x0, v0, theta0, w0, alpha0
valores_iniciales = [0, 0 , 0, 0, 0.1] 

variable_inicial = variable_inicial(valores_iniciales, parametros)


# Optimización de LSSVM
def OptimalControlbyLSSVM(variable_inicial, funcion_objetivo, kernel, valores_iniciales ,parametros):
    sol = minimize(lambda vars: funcion_objetivo(vars, kernel, valores_iniciales, parametros), variable_inicial, method='SLSQP')
    return sol

# Ejecución de la optimización
result = OptimalControlbyLSSVM(variable_inicial, funcion_objetivo, kernel, valores_iniciales, parametros)
print(result)


plot(result, valores_iniciales, parametros)