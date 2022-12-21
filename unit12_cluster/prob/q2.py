from sklearn.cluster import KMeans
import numpy as np

def outlier_detect(Xtr, Xts, nc, t): 
    km = KMeans(n_clusters=nc)
    km.fit(Xtr)
    center = km.cluster_centers_
    center = center[:, :, None].T
    Xts = Xts.reshape(Xts.shape[0], Xts.shape[1], 1)
    dist = (Xts - center) ** 2
    dist = np.sum(dist, axis=1)
    outlier = dist < t
    outlier = np.sum(outlier, axis=1)
    out_index = np.where(outlier == 0)
    outlier = np.zeros(Xts.shape[0])
    outlier[out_index] = 1
    return outlier

Xtr = np.array([
                [1, 2], 
                [2, 4], 
                [1, 0], 
                [0, 2], 
                [10, 2], 
                [-9, 0], 
                [4, 20]])
Xts = np.array([
                [5, 10], 
                [2, 0], 
                [9, 1], 
                [-2, 4], 
                [6, 9]])       
nc = 4
t = 5
# outlier_detect(Xtr, Xts, nc, t)
a = np.array([
            [0, 0],
            [1, 1], 
            [2, 2],
            [3, 3]])
a = a[:, :, None]
a = a.T
b = np.arange(6).reshape(3, 2, 1)
c = a - b
c = c ** 2
c = np.sum(c, axis=1)
c = c > 2
c = np.sum(c, axis=1)
d = np.where(c == 2)
e = np.zeros(c.shape)
e[d] = 1
print(outlier_detect(Xtr, Xts, nc, t))
