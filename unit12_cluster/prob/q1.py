import numpy as np
import matplotlib.pyplot as plt

x1 = np.array([0, 1, 0, 2, 2])
x2 = np.array([0, 0, 1, 2, 3])
x = np.array([x1, x2]).T
c1 = np.array([0, 0])
c2 = np.array([1, 0])
bag = [1, 3, 4]
print(np.mean(x[bag], axis=0))
plt.scatter(x1, x2)
plt.grid()
# plt.show()