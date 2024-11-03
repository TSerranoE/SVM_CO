import numpy as np
import control

sigma = 1
h = 0.05
m = 0.1
m_t = 1.1
largo = 0.5
g = 9.81
Q = np.eye(4)
R = np.array([[0.01]])
A = np.array([[0, 1, 0, 0], [0, 0, -g*m/(4/3*m_t-m), 0], [0, 0, 0, 1], [0, 0, m_t*g/(largo*(4/3*m_t-m)), 0]])
B = np.array([[0], [4/3*(1/(4/3*m_t-m))], [0], [-1/(largo*(4/3*m_t-m))]])

K, S, E = control.lqr(A, B, Q, R)   

print(K)