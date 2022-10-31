import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([30, 50, 70, 80, 100])
x2 = np.array([0, 1, 1, 2, 1])
y = np.array([0, 1, 0, 1, 1])

a1 = 1/200
a2 = 1/200
b = 0.2
z = a1 * x1 + a2 * x2 + b
plt.scatter(x1, y)
plt.scatter(x2, y)
plt.plot(x1, z)
plt.xlabel('Income or Num websites')
plt.ylabel('Donate')
plt.legend(['Income vs Donate', 'Num websites vs Donate'])
plt.grid()
plt.show()