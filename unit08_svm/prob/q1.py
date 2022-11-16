import numpy as np
import matplotlib.pyplot as plt
import math

x1 = np.array([0, 1, 1, 2])
x2 = np.array([0, 0.3, 0.7, 1])
y = np.array([-1, -1, 1, 1])

x = np.linspace(-3, 4, 2)
z = (-0.2 -x1) / -2.4
m = 0.2 / (np.sqrt(1 + 2.4 ** 2))

plt.scatter(x1[:2], x2[:2])
plt.scatter(x1[2:], x2[2:])
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(['y = -1', 'y = 1'])

plt.plot(x1, z)
plt.plot(x1, z + m, '--')
plt.plot(x1, z - m, '--')
plt.grid()
plt.show()
print()