import numpy as np
import matplotlib.pyplot as plt


def F(x, parametros):
    """
    Función de estado del sistema del péndulo invertido.
    x: vector de estado [x1, x2, x3, x4]
    parametros: lista de parámetros del sistema [m, mt, l, g]
    """
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros

    # Variables de estado
    x1, x2, x3, x4 = x
    # Ecuaciones de movimiento
    dx1 = x2
    dx2 = (4/3 * m * largo * x4**2 * np.sin(x3) - m * g * np.sin(2 * x3) / 2) / (4/3 * m_t - m * np.cos(x3)**2)
    dx3 = x4
    dx4 = (m_t * g * np.sin(x3) - m * largo * x4**2 * np.sin(2 * x3) / 2) / (largo * (4/3 * m_t - m * np.cos(x3)**2))

    return np.array([dx1, dx2, dx3, dx4])

def G(x, parametros):
    """
    Función de control del sistema del péndulo invertido.
    x: vector de estado [x1, x2, x3, x4]
    parametros: lista de parámetros del sistema [m, mt, l, g]
    """
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros

    # Variables de estado
    x3 = x[2]
    # Función de control
    g1 = 0
    g2 = 4/3 * (1 / (4/3 * m_t - m * np.cos(x3)**2))
    g3 = 0
    g4 = - np.cos(x3) / (largo * (4/3 * m_t - m * np.cos(x3)**2))

    return np.array([g1, g2, g3, g4])

def kutta(x, x_gorro, alpha, sigma, kernel, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros
    L_laplaciano = -sum((alpha[l]*kernel(x, np.zeros((N, 4)), l, 0, sigma, parametros) /sigma**2)*x[l] for l in range(N))
    u = np.zeros(N)
    for i in range(N-1):
        u[i] = (K-L_laplaciano) @ x_gorro[i] + sum(alpha[l] * kernel(x, x_gorro, l, i, sigma, parametros) for l in range(N))
        k1 = F(x_gorro[i], parametros) + G(x_gorro[i], parametros) * u[i]
        k2 = F(x_gorro[i] + (h/2)*k1, parametros) + G(x_gorro[i] + (h/2)*k1, parametros) * (u[i] + (h/2))
        k3 = F(x_gorro[i] + (h/2)*k2, parametros) + G(x_gorro[i] + (h/2)*k2, parametros) * (u[i] + (h/2))
        k4 = F(x_gorro[i] + h*k3, parametros) + G(x_gorro[i] + h*k3, parametros) * (u[i] + h)
        x_gorro[i+1] = x_gorro[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    u[N-1] = (K-L_laplaciano) @ x_gorro[N-1] + sum(alpha[l] * kernel(x, x_gorro, l, N-1, sigma, parametros) for l in range(N))

    return x_gorro, u

# Definición de la función objetivo
def funcion_objetivo(vars, kernel, valores_iniciales, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros

    # Extraer las variables
    x = vars[:4*N].reshape(4, N).T
    # Modularizar las variables x[N*2:N*3] modulo 2*np.pi
    alpha = vars[4*N:5*N]
    sigma = vars[5*N]

    # Creación de x_gorro
    x_gorro = np.zeros((N, 4))
    x_gorro[0] = valores_iniciales[:4]

    # Cálculo de la función objetivo
    u = np.zeros(N)
    for i in range(N-1):
        u[i] =  sum(alpha[l] * kernel(x, x_gorro, l, i, sigma, parametros) for l in range(N))
        k1 = F(x_gorro[i], parametros) + G(x_gorro[i], parametros) * u[i]
        k2 = F(x_gorro[i] + (h/2)*k1, parametros) + G(x_gorro[i] + (h/2)*k1, parametros) * (u[i] + (h/2))
        k3 = F(x_gorro[i] + (h/2)*k2, parametros) + G(x_gorro[i] + (h/2)*k2, parametros) * (u[i] + (h/2))
        k4 = F(x_gorro[i] + h*k3, parametros) + G(x_gorro[i] + h*k3, parametros) * (u[i] + h)
        x_gorro[i+1] = x_gorro[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4)
    u[N-1] =  sum(alpha[l] * kernel(x, x_gorro, l, N-1, sigma, parametros) for l in range(N))
    # Imprimir el valor de la función objetivo
    objetivo = sum(np.matmul(x_gorro[i].T, x_gorro[i])  for i in range(N-5,N)) + sum(lambda_value*alpha[i]**2 for i in range(N))
    print("valor" , str(objetivo) )
    return (objetivo)



# Definición variables valores_iniciales
def variable_inicial(valores_iniciales, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros
    variable_inicial = np.zeros((N, 5))
    variable_inicial[0] = valores_iniciales[:5]
    variable = np.zeros(N*5+1)
    variable[:N*5]=variable_inicial.T.flatten()
    variable[-1]=valores_iniciales[5]
    return variable

# Definición del kernel en este caso RBF
def kernel(x, x_gorro, l, i, sigma, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros
    return np.exp(-(1/(sigma**2))  *  ((x[l][0]-x_gorro[i][0])**2+(x[l][1]-x_gorro[i][1])**2+(x[l][2]-x_gorro[i][2])**2+(x[l][3]-x_gorro[i][3])**2)) 

# Graficar las soluciones
def plot(result, valores_iniciales, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros

    # Extraer las variables
    x_sol = result.x[:4*N].reshape(4, N).T
    alpha_sol = result.x[4*N:5*N]
    sigma = result.x[5*N]

    # Creación de x_gorro
    x_gorro = np.zeros((N, 4))
    x_gorro[0] = valores_iniciales[:4]

    x_gorro, u_opt = kutta(x_sol, x_gorro, alpha_sol, sigma, kernel, parametros)
    # x_modularizado = x_gorro[N-1]
    # x_modularizado[2] = x_modularizado[2]%(2*np.pi)
    
    # Imprimir el valor de la función objetivo
    objetivo = sum(np.matmul(x_gorro[i].T, x_gorro[i])  for i in range(N-5,N)) + sum(lambda_value*alpha_sol[i]**2 for i in range(N))

    print("valor", objetivo)
    print("posición final", x_gorro[N-1])

    # Graficar las soluciones
    plt.figure(figsize=(12, 12))

    # Posición x
    plt.subplot(5, 1, 1)
    plt.plot(np.linspace(0, 5, N), x_gorro[:, 0], label='Posición x')
    plt.xlabel('Tiempo')
    plt.ylabel('Posición x')
    plt.legend()
    plt.grid()

    # Velocidad v
    plt.subplot(5, 1, 2)
    plt.plot(np.linspace(0, 5, N), x_gorro[:, 1], label='Velocidad v', color='orange')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad v')
    plt.legend()
    plt.grid()

    # Angulo theta
    plt.subplot(5, 1, 3)
    plt.plot(np.linspace(0, 5, N), x_gorro[:, 2], label='Angulo theta')
    plt.xlabel('Tiempo')
    plt.ylabel('Angulo theta')
    plt.legend()
    plt.grid()

    # Velocidad angular w
    plt.subplot(5, 1, 4)
    plt.plot(np.linspace(0, 5, N), x_gorro[:, 3], label='Velocidad angular w')
    plt.xlabel('Tiempo')
    plt.ylabel('Velocidad angular w')
    plt.legend()
    plt.grid()

    # Control
    plt.subplot(5, 1, 5)
    plt.plot(np.linspace(0, 5, N), u_opt, label='Control Óptimo')
    plt.xlabel('Tiempo')
    plt.ylabel('Control')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
