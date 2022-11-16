import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1.3, 2.1, 2.8, 4.2, 5.7])
y = np.array([-1, -1, -1, 1, -1, 1])
t = np.linspace(0, 5, 100)
t = t[:, None]
t = 4
z = x - t
yhat_bool = z > 0
yhat = 2 * yhat_bool - 1
hinge = 1 - y * z
hinge = np.maximum(0, hinge)
print(z)

print(hinge)
# J = np.sum(hinge, axis=1)
print(J.shape)

plt.plot(t, J)
plt.xlabel('t')
plt.ylabel('J(t)')
plt.grid()
plt.show()