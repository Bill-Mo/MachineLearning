import numpy as np
from sklearn.cluster import KMeans
import random

def K_center(Xtr, K): 
    center_index = random.sample(range(Xtr.shape[0]), K)
    return Xtr[center_index]

x = np.arange(20).reshape(10, 2)
print(K_center(x, 4))