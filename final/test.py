import numpy as np
a = np.arange(12).reshape(6, 2)
print(a @ a.T)
print(a.T @ a)