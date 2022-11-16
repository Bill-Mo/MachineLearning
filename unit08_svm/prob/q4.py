import numpy as np
import matplotlib.pyplot as plt

x = np.array([0, 1, 2, 3])
y = np.array([1, -1, 1, -1])
gamma = 3
alpha = np.array([0, 0, 1, 1])
gamma = 0.3
alpha = np.array([1, 1, 1, 1])

x_diff = x - x[:, None]
K = np.exp(-gamma * (x_diff) ** 2)
z = np.sum(alpha * y * K, axis=1)
z = z.reshape(-1, 1)
yhat_bool = z > 0
yhat = 2 * yhat_bool - 1

plt.scatter(x, z)
plt.scatter(x, yhat)
plt.xlabel('x')
plt.ylabel('Prediction')
plt.legend(['z vs x', 'yhat vs x'])
plt.grid()
plt.show()