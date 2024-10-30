import numpy as np
import matplotlib.pyplot as plt
import control as ct


# Definición de la función objetivo
def funcion_objetivo(vars, kernel, valores_iniciales, parametros):
    T, N, lambda_value, sigma, h, m, m_t, largo, g, Q, R, A , B = parametros

    # LQR
    K, S, E = ct.lqr(A, B, Q, R)

    # Extraer las variables
    x = vars[:4*N].reshape(4, N).T
    alpha = vars[4*N:]

    # Creación de x_gorro
    x_gorro = np.zeros((N, 4))
    x_gorro[0] = valores_iniciales[:4]
    u = np.zeros((N, 4))
    for i in range(N-1):
        k1 = A @ x_gorro[i] + B @ u[i]
        k2 = A @ (x_gorro[i] + (h/2)*k1) + B @ (u[i] + (h/2)*k1)
        k3 = A @ (x_gorro[i] + (h/2)*k2) + B @ (u[i] + (h/2)*k2)
        k4 = A @ (x_gorro[i] + h*k3) + B @ (u[i] + h*k3)

        u[i] = np.array([sum(alpha[l] * kernel(x, x_gorro, l, i, j, parametros) for l in range(N)) for j in range(4)])
        x_gorro[i+1] = x_gorro[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    u[N-1] = np.array([sum(alpha[l] * kernel(x, x_gorro, l, N-1, j, parametros) for l in range(N)) for j in range(4)])
    
    # Imprimir el valor de la función objetivo
    objetivo = sum(x_gorro[i].T @ Q @ x_gorro[i] + R*u[i].T @ u[i] + lambda_value * alpha[i]**2 for i in range(N))
    print("valor" , objetivo )
    return (objetivo)

# Definición variables valores_iniciales
def variable_inicial(valores_iniciales, parametros):
    T, N, lambda_value, sigma, h, m, m_t, largo, g, Q, R, A , B = parametros
    variable_inicial = np.zeros((N, 5))
    variable_inicial[0] = valores_iniciales
    return variable_inicial.T.flatten()

# Definición del kernel en este caso RBF
def kernel(x, x_gorro, l, i, index, parametros):
    T, N, lambda_value, sigma, h, m, m_t, largo, g, Q, R, A , B = parametros
    return np.exp(-(1/(2*sigma**2))  *  ((x[l][index]-x_gorro[i][index])**2)) 

# Graficar las soluciones
def plot(result, valores_iniciales, parametros):
    T, N, lambda_value, sigma, h, m, m_t, largo, g, Q, R, A , B = parametros


    # Extraer las soluciones
    x_sol = result.x[:4*N].reshape(4, N).T
    alpha_sol = result.x[4*N:]


    x_gorro = np.zeros((N, 4))
    x_gorro[0] = valores_iniciales[:4]
    u_opt = np.zeros((N, 4))
    for i in range(N-1):
        k1 = A @ x_gorro[i] + B @ u_opt[i]
        k2 = A @ (x_gorro[i] + (h/2)*k1) + B @ (u_opt[i] + (h/2)*k1)
        k3 = A @ (x_gorro[i] + (h/2)*k2) + B @ (u_opt[i] + (h/2)*k2)
        k4 = A @ (x_gorro[i] + h*k3) + B @ (u_opt[i] + h*k3)

        u_opt[i] = np.array([sum(alpha_sol[l] * kernel(x_sol, x_gorro, l, i, j, parametros) for l in range(N)) for j in range(4)])
        x_gorro[i+1] = x_gorro[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    u_opt[N-1] = np.array([sum(alpha_sol[l] * kernel(x_sol, x_gorro, l, N-1, j, parametros) for l in range(N)) for j in range(4)])
    
    # Imprimir el valor de la función objetivo
    objetivo = sum(x_gorro[i].T @ Q @ x_gorro[i] + R*u_opt[i].T @ u_opt[i] + lambda_value * alpha_sol[i]**2 for i in range(N))   

    # Imprimir el valor de la función objetivo    
    print("valor", objetivo)
    print("posición final", x_gorro[N-1])

    # Graficar las soluciones
    plt.figure(figsize=(12, 12))

    # Posición x
    plt.subplot(5, 1, 1)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 0], label='Posición x')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición x')
    plt.legend()
    plt.grid()

    # Velocidad v
    plt.subplot(5, 1, 2)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 1], label='Velocidad v', color='orange')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad v')
    plt.legend()
    plt.grid()

    # Angulo theta
    plt.subplot(5, 1, 3)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 2], label='Angulo theta')
    plt.xlabel('Tiempo')
    plt.ylabel('Angulo theta')
    plt.legend()
    plt.grid()

    # Velocidad angular w
    plt.subplot(5, 1, 4)
    plt.plot(np.linspace(0, T, N), x_gorro[:, 3], label='Velocidad angular w')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad angular w')
    plt.legend()
    plt.grid()

    # Control
    plt.subplot(5, 1, 5)
    plt.plot(np.linspace(0, T, N), u_opt, label='Control Óptimo')
    plt.xlabel('Tiempo')
    plt.ylabel('Control')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
