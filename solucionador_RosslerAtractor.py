from Ej2_RosslerAtractor import funcion_objetivo, variable_inicial, kernel, plot
from scipy.optimize import minimize


#Constantes
T = 10
N = 50
lambda_value = 0.01
sigma = 1
a = 0.1
b = 2
c = 4


# Parámetros
parametros = [T, N, lambda_value, sigma, a, b, c]

# x0, y0, z0, alpha0
valores_iniciales = [1, -1, 0, 0.1] 

variable_inicial = variable_inicial(valores_iniciales, parametros)


# Optimización de LSSVM
def OptimalControlbyLSSVM(variable_inicial, funcion_objetivo, kernel, valores_iniciales ,parametros):
    sol = minimize(lambda vars: funcion_objetivo(vars, kernel, valores_iniciales, parametros), variable_inicial, method='SLSQP')
    return sol

# Ejecución de la optimización
result = OptimalControlbyLSSVM(variable_inicial, funcion_objetivo, kernel, valores_iniciales, parametros)
print(result)


plot(result, valores_iniciales, parametros)