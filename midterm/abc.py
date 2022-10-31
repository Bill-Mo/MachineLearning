import numpy as np
from sklearn.linear_model import Ridge

a = np.array([
    [2, 2, 1, 1, 1], 
    [2, 5, 1, 8, 40]])
print(a@a)
Ridge