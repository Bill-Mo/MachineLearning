import sklearn.linear_model as lm
import numpy as np
import matplotlib.pyplot as plt

# x1 = np.array([0, 0, 1, 1])
# x2 = np.array([0, 1, 0, 1])
# x = np.array([x1, x2])
# x = x.T
# y = np.array([1, 4, 3, 7])

# model = lm.LinearRegression()
# print('{}\n{}'.format(x.shape, y.shape))
# model.fit(x, y)
# beta0 = model.intercept_
# beta1 = model.coef_[0]
# beta2 = model.coef_[1]
# print(beta0)
# print(beta1, beta2)

# y_pred = beta0 + beta1 * x1 + beta2 * x2
# print(y_pred)

# a
# x = np.insert(x, 0, 1, axis=1)
# print(x)
# beta = np.array([beta0, beta1, beta2])
# n = x.shape[0]
# yhat = np.zeros(n)
# for i in range(n):
#     yhat[i] = beta[0] * x[i ,0] + beta[1] * x[i ,1] + beta[2] * x[i,1] * x[i ,2]
# print(yhat)

# x[:, 2] = x[:, 1] * x[:, 2]
# yhat = x.dot(beta)
# print(yhat)

# b
# x = np.array([1, 2, 4, 6, 7, 10, 19])
# alpha = np.array([1, 3, 9])
# beta = np.array([beta0, beta1, beta2])
# n = len(x)
# m = len(alpha)
# yhat = np.zeros(n)
# for i in range(n):
#     for j in range(m):
#         yhat[i] += alpha[j]*np.exp(-beta[j] * x[i])
# print(yhat)

# x = np.reshape(x, (1, -1))
# alpha = np.reshape(alpha, (1, -1))
# beta = np.reshape(beta, (-1, 1))
# m = np.dot(-beta, x)
# yhat = alpha.dot(np.exp(m))
# print(yhat)

# c
x = np.array([[1, 2, 3], 
              [1, 4, 5], 
              [4, 2, 1], 
              [11, 222, 3], 
              [0, 0, 1]])

y = np.array([[1, 26, 1], 
              [90, 9, 8]])
n, d = x.shape
m, d = y.shape
dist = np.zeros((n, m))
for i in range(n): 
    for j in range(m): 
        for k in range(d): 
            dist[i, j] += (x[i, k] - y[j, k]) ** 2
print(dist)

x = np.expand_dims(x, 1).repeat(m, axis=1)
y = np.expand_dims(y, 0).repeat(n, axis=0)
dist = x - y
dist = dist ** 2
dist = dist.sum(axis=-1)
print(dist)