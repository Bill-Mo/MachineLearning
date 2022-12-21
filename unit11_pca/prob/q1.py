import numpy as np
from sklearn.decomposition import PCA
from numpy.linalg import svd

x = np.array([
    [3, 2, 1], 
    [2, 4, 5], 
    [1, 2, 3], 
    [0, 2, 5],
])
mean = np.mean(x, axis=0)
cov = np.zeros((3, 3))
for i in range(3): 
    for j in range(3):
        for n in range(4):  
            cov[i, j] += (x[n, i] - mean[i]) * (x[n, j] - mean[j])
cov = cov / 4
eigvalue, eigvector = np.linalg.eig(cov)

Z = (x - mean).dot(eigvector)

xhat = Z.dot(eigvector.T) + mean

Z_2 = Z[:, :2].dot(eigvector[:, :2].T) + mean

e1 = np.mean(np.sum((Z_2 - x) ** 2, axis=1))
# print(mean)
print(cov)
print(eigvalue, eigvector)
print(Z)
print(xhat)
print(Z_2)
print(e1, eigvalue[-1])
