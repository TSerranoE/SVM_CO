import numpy as np
import control

# Parámetros del sistema
m = 0.1
mt = 1.1
l = 0.5
g = 9.81

# Matrices del sistema linealizado
A = np.array([[0, 1, 0, 0],
              [0, 0, -m*g/(4/3*mt-m), 0],
              [0, 0, 0, 1],
              [0, 0, mt*g/(l*(4/3*mt-m)), 0]])

B = np.array([[0],
              [4/3 / (4/3*mt - m)],
              [0],
              [-1 / (l*(4/3*mt - m))]])

# Matrices de costo
Q = np.eye(4) # Matriz identidad 4x4
R = 0.01      # Escalar

# Cálculo del controlador LQR
K, S, E = control.lqr(A, B, Q, R)

# Mostrar la matriz de retroalimentación resultante
L_lqr = K
print(L_lqr)
