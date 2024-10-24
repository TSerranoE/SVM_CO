import numpy as np
import matplotlib.pyplot as plt

x_ref = []

for k in range(100):
    x_ref.append([np.sin(2*np.pi*k/20), np.cos(2*np.pi*k/20)])

# Convertir a array para manejo más sencillo
x_ref = np.array(x_ref)

# Extraer las columnas correspondientes a las coordenadas x (sin) y y (cos)
x_values = x_ref[:, 0]
y_values = x_ref[:, 1]

# Graficar
plt.plot(x_values, y_values)
plt.title('Plot of sin and cos values')
plt.xlabel('sin(2*pi*k/20)')
plt.ylabel('cos(2*pi*k/20)')
plt.grid(True)
plt.axis('equal')  # Para asegurar que los ejes tengan la misma escala
plt.show()
