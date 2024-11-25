import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Definición variables valores_iniciales
def variable_inicial(valores_iniciales, parametros):
    _, N, _, _, _, _ = parametros
    variable_inicial = np.zeros((N, 3))
    variable_inicial[0] = valores_iniciales
    return variable_inicial.T.flatten()

# Definición del kernel en este caso RBF
def kernel(x, x_gorro, l, i, parametros):
    _, _, _, _, sigma, _ = parametros
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][0]-x_gorro[i][0])**2 + (x[l][1]-x_gorro[i][1])**2))

# Definición de la función objetivo
def funcion_objetivo(vars, kernel, valores_iniciales, parametros):
    T, N, vf, lambda_value, _, gamma = parametros
    # Extraer las variables
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]
    # Imprimir el valor de la función objetivo
    print("valor", (x[N-1][1]-vf)**2 + lambda_value*np.sum(alpha**2))
    return (x[N-1][1]-vf)**2 + lambda_value*np.sum(alpha**2)



T = 10
N = 30
vf = 100
lambda_value = 0.01
sigma = 1
gamma = 1

# Parámetros
parametros = [T, N, vf, lambda_value, sigma, gamma]

# x0, v0, alpha0
valores_iniciales = [80, 80, 0.1] 

variable_inicial = variable_inicial(valores_iniciales, parametros)


# Definir la restricción
def constraint(vars):
    x = vars[:2*N].reshape(2, N).T
    alpha = vars[2*N:]
    constraints = []
    constraints.append(x[0][0] - valores_iniciales[0])
    constraints.append(x[0][1] - valores_iniciales[1])  
    for i in range(N-1):
        constraints.append(x[i+1][0] -  x[i][0] + (T/N)*(x[i][1]) )
        constraints.append(x[i+1][1] - x[i][1] + (T/N)*(-gamma*x[i][0] + x[i][1] + sum(alpha[l]*kernel(x, x, l, i, parametros) for l in range(N))))
    return np.array(constraints)

# Definir las restricciones en el formato requerido por scipy.optimize.minimize
constraints = [{'type': 'eq', 'fun': constraint}]

# Minimizar la función objetivo con restricciones
result = minimize(lambda vars: funcion_objetivo(vars, kernel, valores_iniciales, parametros), 
                  variable_inicial, method='SLSQP', constraints=constraints)

# Ejecución de la optimización
print("primera optimización", result)



# Graficar las soluciones
def plot(result, valores_iniciales, parametros):
    T, N, vf, lambda_value, _, gamma = parametros

    # Extraer las soluciones
    x_sol = result.x[:2*N].reshape(2, N).T
    alpha_sol = result.x[2*N:]

    # creación de x_gorro
    x_gorro = np.zeros((N, 2))
    x_gorro[0] = valores_iniciales[:2]

    # Definición del control
    u_opt = np.zeros(N)

    for i in range(N-1):
        x_gorro[i+1][0] =  x_gorro[i][0] + (T/N)*(x_gorro[i][1]) 
        x_gorro[i+1][1] = x_gorro[i][1] + (T/N)*(-gamma*x_gorro[i][0] + x_gorro[i][1] + sum(alpha_sol[l]*
                                                        kernel(x_sol, x_gorro, l, i, parametros) for l in range(N))) 
    
    for i in range(N):
       u_opt[i] = sum(alpha_sol[l]* kernel(x_sol, x_gorro, l, i, parametros) for l in range(N))

    # Imprimir el valor de la función objetivo    
    print("valor", (x_gorro[N-1][1]-vf)**2 )
    print("velocidad final", x_gorro[N-1][1])

    # Graficar las soluciones
    plt.figure(figsize=(12, 9))

    # Posición x
    plt.subplot(3, 1, 1)
    plt.plot(np.linspace(0,T,N), x_gorro[:, 0], label='Posición x')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición x')
    plt.legend()
    plt.grid()

    # Velocidad v
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0,T,N), x_gorro[:, 1], label='Velocidad v', color='orange')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad v')
    plt.legend()
    plt.grid()

    # Control
    plt.subplot(3, 1, 3)
    plt.plot(np.linspace(0,T,N), u_opt, label='Control Óptimo')
    plt.xlabel('Tiempo')
    plt.ylabel('Control')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

plot(result, valores_iniciales, parametros)