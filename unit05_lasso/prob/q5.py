import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

# Dummy value
a = 10**-2
b = 10**2
lam = 5
X = np.random.rand(200, 100)
y = np.arange(200)
print(X.shape, y.shape)
xtr, xts, ytr, yts = train_test_split(X, y, test_size=0.33)


p = np.linspace(a, b, 100)
Ztr = np.exp(-p*xtr)
Zts = np.exp(-p*xts)
model = Lasso(lam).fit(Ztr, ytr)
beta = model.coef_
yhat = model.predict(Zts)
rss = np.sum((yts-yhat)**2)
print(rss)
rank = np.argsort(beta)
best = rank[-3:]
print('Best 3 alpha: {}'.format(p[best]))
print('Best 3 beta: {}'.format(beta[best]))