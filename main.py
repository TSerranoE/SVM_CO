import numpy as np
from scipy.optimize import minimize


def kernel(x_l, x_i, kernel_func):
    return kernel_func(x_l, x_i)


def cost_function(alpha, x, lambda_value, kernel_func, f, J_N):
    N = len(x)
    u = np.zeros(N-1)
    

    for i in range(N-1):
        u[i] = sum(alpha[l] * kernel(x[l], x[i], kernel_func) for l in range(N))
    

    x_new = np.zeros(N)
    x_new[0] = x[0] 
    for i in range(1, N):
        x_new[i] = f(x_new[i-1], u[i-1])


    J = J_N(x_new, u) + lambda_value * sum(alpha[i]**2 for i in range(N-1))
    
    return J


def solve_optimal_control(x_initial, lambda_value, kernel_func, f, J_N):
    N = len(x_initial)
    alpha_initial = np.random.randn(N-1) 
    

    result = minimize(cost_function, alpha_initial, args=(x_initial, lambda_value, kernel_func, f, J_N))
    
    return result.x  


x_initial = np.array([1.0, 2.0, 3.0, 4.0])  
lambda_value = 0.1 


def linear_kernel(x_l, x_i):
    return np.dot(x_l, x_i)


def f(xi, ui):
    return xi + ui


def J_N(x, u):
    return sum(xi**2 + ui**2 for xi, ui in zip(x, u))


optimal_alpha = solve_optimal_control(x_initial, lambda_value, linear_kernel, f, J_N)
print("Valores óptimos de alpha:", optimal_alpha)
