from sklearn.preprocessing import StandardScaler
import numpy as np
from numpy.linalg import svd
X = np.arange(2000).reshape(200, 10)

scaling = StandardScaler()
scaling.fit(X)
Z = scaling.transform(X)

U, S, Vtr = svd(Z, full_matrices = False)

# i)
PCs = Vtr
mean = np.mean(X, axis=0)

# ii)
PoV_len = S.shape[0]
lam = S**2
PoV = np.cumsum(lam)/np.sum(lam)
min_PoV = np.min(np.where(PoV >= 0.9)) + 1

# iii)
Z = U[:,:min_PoV]*S[None,:min_PoV]
W = PCs[:min_PoV, :]
print(Z.shape)
Xhat = mean + Z.dot(W)