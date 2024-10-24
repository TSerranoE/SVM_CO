from ejemplo_creado import funcion_objetivo, variable_inicial, kernel, plot
from scipy.optimize import minimize
import timeout_decorator # type: ignore

#Constantes
T = 10
N = 70
vf = 100
lambda_value = 0
sigma = 1
gamma = 1

# Parámetros
parametros = [T, N, vf, lambda_value, sigma, gamma]

# x0, v0, alpha0
valores_iniciales = [3, 100, 1] 

variable_inicial = variable_inicial(valores_iniciales, parametros)


# Optimización de LSSVM
tiempo_maximo = 300
@timeout_decorator.timeout(tiempo_maximo) 
def OptimalControlbyLSSVM(variable_inicial, funcion_objetivo, kernel, valores_iniciales ,parametros):
    sol = minimize(lambda vars: funcion_objetivo(vars, kernel, valores_iniciales, parametros), variable_inicial, method='SLSQP')
    return sol

# Ejecución de la optimización

try:
    result = OptimalControlbyLSSVM(variable_inicial, funcion_objetivo, kernel, valores_iniciales, parametros)
    print(result)
except timeout_decorator.timeout_decorator.TimeoutError:
    print("La optimización se detuvo después de ${tiempo_maximo} segundos.")

plot(result, valores_iniciales, parametros)