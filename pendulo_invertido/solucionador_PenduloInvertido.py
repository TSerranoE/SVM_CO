import numpy as np
from Ej3_PenduloInvertido import funcion_objetivo, variable_inicial, kernel, plot
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_are

# Constantes
T = 5
N = 100
lambda_value = 0
h = 0.05
m = 0.1
m_t = 1.1
largo = 0.5
g = 9.81

A = np.array([[0, 1, 0, 0], 
              [0, 0, -g*m/(4/3*m_t-m), 0], 
              [0, 0, 0, 1], 
              [0, 0, m_t*g/(largo*(4/3*m_t-m)), 0]])

B = np.array([[0], 
              [4/3*(1/(4/3*m_t-m))], 
              [0], 
              [-1/(largo*(4/3*m_t-m))]])

Q = np.eye(4)  # 4x4 Identity matrix
R = 0.01*np.diag([1])  # 1x1 Diagonal matrix

# Solve the continuous-time algebraic Riccati equation
P = solve_continuous_are(A, B, Q, R)

# Compute the LQR gain
K = np.linalg.inv(R) @ B.T @ P

B = np.array([0, 4/3*(1/(4/3*m_t-m)), 0, -1/(largo*(4/3*m_t-m))])
R = 0.01
# Parámetros
parametros = [T, N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K[0]]

# x0, v0, theta0, w0, alpha0, sigma0
valores_iniciales = [-0.02, 0.01, 0, 0.02, 0.1, np.sqrt(10)] 

variables_iniciales = variable_inicial(valores_iniciales, parametros)


# Optimización de LSSVM
def OptimalControlbyLSSVM(variables_iniciales, funcion_objetivo, kernel, valores_iniciales ,parametros):
    sol = minimize(lambda vars: funcion_objetivo(vars, kernel, valores_iniciales, parametros), 
                   variables_iniciales, method='SLSQP')
    return sol

# Ejecución de la optimización
result = OptimalControlbyLSSVM(variables_iniciales, funcion_objetivo, kernel, valores_iniciales, parametros)
print(result)

plot(result, valores_iniciales, parametros)