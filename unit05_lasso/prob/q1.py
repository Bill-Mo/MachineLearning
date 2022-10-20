from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np

X = np.array([[1, 2, 3, 4], 
              [2, 3, 5, 11], 
              [21, 444, 11, 2], 
              [0, 1, 0, 1], 
              [0, 10, -2, -55], 
              [2, 4, 5, 6], 
              [1, 1, 1, 1], 
              [3, 4, 0, 0]])

y = np.array([[1, 2],
              [11, 54], 
              [2, 0], 
              [-3, -3], 
              [11111, 22], 
              [4, 5], 
              [0, 1], 
              [33, -100]])
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.33)

# a
row_tr, col = Xtr.shape
row_ts, col = Xts.shape
Xtr_f = np.zeros((row_tr, 1))
Xts_f = np.zeros((row_ts, 1))
r2s = np.zeros(col)
for f in range(col): 
    Xtr_f = Xtr[:, f].reshape(-1, 1)
    Xts_f = Xts[:, f].reshape(-1, 1)
    model = LinearRegression().fit(Xtr_f, ytr)
    yhat = model.predict(Xts_f)
    r2s[f] = r2_score(yts, yhat)
print('best feature: ', np.argmax(r2s))
print('best r2: ', np.max(r2s))

# b
row_tr, col = Xtr.shape
row_ts, col = Xts.shape
Xtr_f = np.zeros((row_tr, 2))
Xts_f = np.zeros((row_ts, 2))
r2s = np.zeros((col, col))
for f1 in range(col): 
    for f2 in range(col): 
        if f1 == f2: 
            r2s[f1, f2] = -np.inf
            continue
        Xtr_f[:, 0] = Xtr[:, f1]
        Xts_f[:, 0] = Xts[:, f1]
        Xtr_f[:, 1] = Xtr[:, f2]
        Xts_f[:, 1] = Xts[:, f2]
        model = LinearRegression().fit(Xtr_f, ytr)
        yhat = model.predict(Xts_f)
        r2s[f1, f2] = r2_score(yts, yhat)
max = np.max(r2s)
max_x, max_y = np.where(r2s == max)
print('best feature 1: ', max_x + 1)
print('best feature 2: ', max_y + 1)
print('best r2: ', max) 