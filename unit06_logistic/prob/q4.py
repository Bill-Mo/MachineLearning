import numpy as np

beta = [-6, 0.05, 1]
x = [1, 40, 3.5]
z = np.matmul(beta, x)
print(z)
p = 1 / (1 + np.exp(-z))
print(p)