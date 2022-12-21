from sklearn.cluster import KMeans
import numpy as np
from sklearn.linear_model import LinearRegression

def preprocessing(Xtr, ytr, Xts, yts, nc): 
    km = KMeans(n_clusters=nc)
    km.fit(Xtr)
    cluster = km.predict(Xtr)

    for c in range(nc):
        Xtr_c = Xtr[cluster == c]
        ytr_c = ytr[cluster == c]
        reg = LinearRegression()
        reg.fit(Xtr_c, ytr_c)
        yhat = reg.predict(Xts)
        mse = np.mean((yts - yhat) ** 2)
        

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
nc = 3
t = 5

preprocessing(Xtr, [], Xts, [], nc)