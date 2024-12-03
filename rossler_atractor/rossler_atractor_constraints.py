import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


#Constantes
T = 10
N = 30
lambda_value = 0.01
sigma = 1
gamma = 0.1
a = 0.1
b = 2
c = 4
ref = [0, -0.5, 0.5]



# Definición variables valores_iniciales
def variable_inicial(valores_iniciales, parametros):
    T, N, lambda_value, sigma, gamma, a, b, c = parametros
    variable_inicial = np.zeros((N, 4))
    variable_inicial[0] = valores_iniciales
    return variable_inicial.T.flatten()

# Parámetros
parametros = [T, N, lambda_value, sigma, gamma, a, b, c]

# x0, y0, z0, alpha0
valores_iniciales = [1, -1, 0, 0.1] 

variable_inicial = variable_inicial(valores_iniciales, parametros)


def kernel(x, x_gorro, l, i, index, parametros):
    T, N, lambda_value, sigma, gamma, a, b, c = parametros
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][index]-x_gorro[i][index])**2)) 

# Definición de la función objetivo
def funcion_objetivo(vars, kernel, ref, valores_iniciales, parametros):
    T, N, lambda_value, sigma, gamma, a, b, c = parametros

    # Extraer las variables
    x = vars[:3*N].reshape(3, N).T
    alpha = vars[3*N:]

    # Creación de x_gorro
    x_gorro = x
    obj = sum( (x_gorro[i][0] - ref[0])**2 + (x_gorro[i][1] - ref[1])**2 + (x_gorro[i][2] - ref[2])**2  
               + gamma*(sum(alpha[l]*(kernel(x, x_gorro, l , i, 0, parametros))for l in range(N))**2
                      + sum(alpha[l]*(kernel(x, x_gorro, l , i, 1, parametros))for l in range(N))**2
                      + sum(alpha[l]*(kernel(x, x_gorro, l , i, 2, parametros))for l in range(N))**2) 
                      for i in range(N)) + lambda_value*np.sum(alpha**2)
    
    # Imprimir el valor de la función objetivo
    print("valor", obj)
    return obj

# Definir la restricción
def constraint(vars):
    x = vars[:3*N].reshape(3, N).T
    alpha = vars[3*N:]

    constraints = []
    constraints.append(x[0][0] - valores_iniciales[0])
    constraints.append(x[0][1] - valores_iniciales[1])
    constraints.append(x[0][2] - valores_iniciales[2])  
    for i in range(N-1):
        constraints.append(x[i+1][0] -  x[i][0] + (T/N)*(-x[i][1] - x[i][2] + sum(alpha[l]*kernel(x, x, l, i, 0,  parametros) for l in range(N)))  )
        constraints.append(x[i+1][1] -  x[i][1] + (T/N)*(x[i][0] - a*x[i][1] + sum(alpha[l]*kernel(x, x, l, i, 1, parametros) for l in range(N))))
        constraints.append(x[i+1][2] -  x[i][2] + (T/N)*(b + x[i][2]*(x[i][0] - c)  + sum(alpha[l]*kernel(x, x, l, i, 2, parametros) for l in range(N))))
    return np.array(constraints)

# Definir las restricciones en el formato requerido por scipy.optimize.minimize
constraints = [{'type': 'eq', 'fun': constraint}]

# Minimizar la función objetivo con restricciones
result = minimize(lambda vars: funcion_objetivo(vars, kernel, ref, valores_iniciales, parametros), 
                  variable_inicial, method='SLSQP', constraints=constraints)

# Ejecución de la optimización
print("primera optimización", result)


# Graficar las soluciones
def plot(result, ref, valores_iniciales, parametros):
    T, N, lambda_value, sigma, gamma, a, b, c = parametros

    # Extraer las soluciones
    x_sol = result.x[:3*N].reshape(3, N).T
    alpha_sol = result.x[3*N:]

    # creación de x_gorro
    x_gorro = x_sol
    # Definición del control
    u_opt = np.zeros((N, 3))  # Cambiar u_opt para almacenar valores en cada dimensión

    for i in range(N):
        u_opt[i, 0] = sum(alpha_sol[l]* kernel(x_sol, x_gorro, l, i, 0, parametros) for l in range(N))
        u_opt[i, 1] = sum(alpha_sol[l]* kernel(x_sol, x_gorro, l, i, 1, parametros) for l in range(N))
        u_opt[i, 2] = sum(alpha_sol[l]* kernel(x_sol, x_gorro, l, i, 2, parametros) for l in range(N))
       
    obj = sum( (x_gorro[i][0] - ref[0])**2 + (x_gorro[i][1] - ref[1])**2 + (x_gorro[i][2] - ref[2])**2  
               + gamma*(sum(alpha_sol[l]*(kernel(x_sol, x_gorro, l, i, 0, parametros))for l in range(N))**2
                      + sum(alpha_sol[l]*(kernel(x_sol, x_gorro, l, i, 1, parametros))for l in range(N))**2
                      + sum(alpha_sol[l]*(kernel(x_sol, x_gorro, l, i, 2, parametros))for l in range(N))**2) 
                      for i in range(N)) 

    # Imprimir el valor de la función objetivo    
    print("valor", obj)
    print("posición final", x_gorro[N-1])

    # Graficar las soluciones
    plt.figure(figsize=(12, 9))

    # Posición x
    plt.subplot(4, 1, 1)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 0], label='Posición x')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición x')
    plt.legend()
    plt.grid()

    # Posición y
    plt.subplot(4, 1, 2)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 1], label='Posición y', color='orange')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición y')
    plt.legend()
    plt.grid()

    # Posición z
    plt.subplot(4, 1, 3)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 2], label='Posición z', color='green')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición z')
    plt.legend()
    plt.grid()

    # Control
    plt.subplot(4, 1, 4)
    plt.plot(np.linspace(0, T, N), u_opt[:, 0], label='Control Óptimo - Componente 0')
    plt.plot(np.linspace(0, T, N), u_opt[:, 1], label='Control Óptimo - Componente 1')
    plt.plot(np.linspace(0, T, N), u_opt[:, 2], label='Control Óptimo - Componente 2')
    plt.xlabel('Tiempo')
    plt.ylabel('Control')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()

plot(result, ref, valores_iniciales, parametros)