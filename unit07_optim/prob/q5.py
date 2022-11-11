import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
x = np.linspace(-5, 5, 1000)
y = (1/4)*x**2+1-np.cos(2*np.pi*x)
# print(y)
plt.plot(x, y)

plt.grid()
plt.show()