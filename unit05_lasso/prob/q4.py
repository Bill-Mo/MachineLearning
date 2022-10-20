from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler

X = np.random.rand(100, 100)
y = np.arange(100)

Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.33)

Xscal = StandardScaler()
yscal = StandardScaler()
Xtr = Xscal.fit_transform(Xtr)
ytr = yscal.fit_transform(ytr[:, None])
Xts = Xscal.transform(Xts)
yts = yscal.transform(yts[:, None])

model = LinearRegression().fit(Xtr, ytr)

yhat = model.predict(Xts)

rss = np.sum((yts-yhat)**2)