import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_continuous_are
import matplotlib.pyplot as plt

# Constantes
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
print(K[0])

B = np.array([0, 4/3*(1/(4/3*m_t-m)), 0, -1/(largo*(4/3*m_t-m))])
R = 0.01
# Parámetros
parametros = [N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K[0]]
valores_iniciales = [0.01, 0.02, -0.01, 0, 0.01, np.sqrt(10)] # x0, v0, theta0, w0, alpha0, sigma0

# Definición variables valores_iniciales
def variable_inicial(valores_iniciales, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros
    variable_inicial = np.random.uniform(-0.1, 0.1, (N, 5))
    variable_inicial[0] = valores_iniciales[:5]
    variable = np.zeros(N*5+1)
    variable[:N*5] = variable_inicial.T.flatten()
    variable[-1] = valores_iniciales[5]
    return variable

variables_iniciales = variable_inicial(valores_iniciales, parametros)

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

# Definición del kernel en este caso RBF
def kernel(x, x_gorro, l, i, sigma, parametros):
    N, lambda_value, h, m, m_t, largo, g, Q, R, A , B, K = parametros
    return np.exp(-(1/(sigma**2))  *  ((x[l][0]-x_gorro[i][0])**2+(x[l][1]-x_gorro[i][1])**2+(x[l][2]-x_gorro[i][2])**2+(x[l][3]-x_gorro[i][3])**2)) 


def funcion_obj(vars):
    x = vars[:4*N].reshape(4, N).T
    valor = sum(np.matmul(x[i].T, x[i]) for i in range(N-6, N))
    print("Valor: ", valor)
    return valor



# Definir la restricción
def constraint(vars):
    x = vars[:4*N].reshape(4, N).T
    alpha = vars[4*N:5*N]
    sigma = vars[5*N]
    constraints = []
    constraints.append(x[0][0] - valores_iniciales[0])
    constraints.append(x[0][1] - valores_iniciales[1])
    constraints.append(x[0][2] - valores_iniciales[2])
    constraints.append(x[0][3] - valores_iniciales[3])
    u = np.zeros(N)
    L_laplaciano = sum((2*alpha[l]*np.exp(-np.matmul(x[l].T, x[l])/sigma**2)/sigma**2)*x[l] for l in range(N))
    for i in range(N-1):
        u[i] =  (K-L_laplaciano) @ x[i] + sum(alpha[l] * kernel(x, x, l, i, sigma, parametros) for l in range(N))
        k1 = F(x[i], parametros) + G(x[i], parametros) * u[i]
        k2 = F(x[i] + (h/2)*k1, parametros) + G(x[i] + (h/2)*k1, parametros) * (u[i] + (h/2))
        k3 = F(x[i] + (h/2)*k2, parametros) + G(x[i] + (h/2)*k2, parametros) * (u[i] + (h/2))
        k4 = F(x[i] + h*k3, parametros) + G(x[i] + h*k3, parametros) * (u[i] + h)
        
        constraints.extend(x[i+1] - x[i] + (h/6)*(k1 + 2*k2 + 2*k3 + k4))
    return np.array(constraints)


# Definir las restricciones en el formato requerido por scipy.optimize.minimize
constraints = [{'type': 'eq', 'fun': constraint},
               {'type': 'ineq', 'fun': lambda vars: vars[5*N]}]  # sigma > 0

# Minimizar la función objetivo con restricciones
result = minimize(funcion_obj, variables_iniciales, method='SLSQP', constraints=constraints)
# Ejecución de la optimización
print("primera optimización", result)



# Cálculo de la función objetivo
# Extraer las variables
x_sol = result.x[:4*N].reshape(4, N).T
alpha_sol = result.x[4*N:5*N]
sigma_sol = result.x[5*N]


u = np.zeros(N)
L_laplaciano = sum((2*alpha_sol[l]*np.exp(-np.matmul(x_sol[l].T, x_sol[l])/sigma_sol**2)/sigma_sol**2)*x_sol[l] for l in range(N))
for i in range(N-1):
    u[i] =  (K-L_laplaciano) @ x_sol[i] + sum(alpha_sol[l] * kernel(x_sol, x_sol, l, i, sigma_sol, parametros) for l in range(N))
    k1 = F(x_sol[i], parametros) + G(x_sol[i], parametros) * u[i]
    k2 = F(x_sol[i] + (h/2)*k1, parametros) + G(x_sol[i] + (h/2)*k1, parametros) * (u[i] + (h/2))
    k3 = F(x_sol[i] + (h/2)*k2, parametros) + G(x_sol[i] + (h/2)*k2, parametros) * (u[i] + (h/2))
    k4 = F(x_sol[i] + h*k3, parametros) + G(x_sol[i] + h*k3, parametros) * (u[i] + h)
u[N-1] =  (K-L_laplaciano) @ x_sol[N-1] + sum(alpha_sol[l] * kernel(x_sol, x_sol, l, N-1, sigma_sol, parametros) for l in range(N))

objetivo = sum(np.matmul(x_sol[i].T, x_sol[i])  for i in range(N-6,N))

print("valor", objetivo)
print("posición final", x_sol[N-1])

# Graficar las soluciones
plt.figure(figsize=(12, 12))

# Posición x
plt.subplot(5, 1, 1)
plt.plot(np.linspace(0, 5, N), x_sol[:, 0], label='Posición x')
plt.xlabel('Tiempo')
plt.ylabel('Posición x')
plt.legend()
plt.grid()

# Velocidad v
plt.subplot(5, 1, 2)
plt.plot(np.linspace(0, 5, N), x_sol[:, 1], label='Velocidad v', color='orange')
plt.xlabel('Tiempo')
plt.ylabel('Velocidad v')
plt.legend()
plt.grid()

# Angulo theta
plt.subplot(5, 1, 3)
plt.plot(np.linspace(0, 5, N), x_sol[:, 2], label='Angulo theta')
plt.xlabel('Tiempo')
plt.ylabel('Angulo theta')
plt.legend()
plt.grid()

# Velocidad angular w
plt.subplot(5, 1, 4)
plt.plot(np.linspace(0, 5, N), x_sol[:, 3], label='Velocidad angular w')
plt.xlabel('Tiempo')
plt.ylabel('Velocidad angular w')
plt.legend()
plt.grid()

# Control
plt.subplot(5, 1, 5)
plt.plot(np.linspace(0, 5, N), u, label='Control Óptimo')
plt.xlabel('Tiempo')
plt.ylabel('Control')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
