import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def plt_face(x):
    h = 50
    w = 37
    plt.imshow(x.reshape((h, w)), cmap=plt.cm.gray)
    plt.xticks([])
    plt.yticks([])
    
X = np.arange(1000 * 28 * 28).reshape(1000, 28, 28)
Y = np.reshape(X, (1000, 28*28))

Y = Y[:500, :]
nc = 100

pca = PCA(n_components=nc)
pca.fit(Y)
Z = pca.transform(Y)
Yhat = pca.inverse_transform(Z)
plt_face(Yhat)

