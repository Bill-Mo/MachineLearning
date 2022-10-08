import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# a = np.array([
#               [1, 2, 3, 4], 
#               [12, 11, 35, 1], 
#               [0, 0, 2, 0], 
#               [1, 2, 3, 4], 
#               [12, 11, 35, 1], 
#               [0, 0, 2, 0]
#               ])
# b = np.zeros((3, 8))
# new_line = np.zeros(3)
# for i in range(3): 
#     b[i, :] = np.array([a[i, :], a[i + 1, :]]).reshape(1, -1)
# print(b)
# print(b.reshape(1, -1))
# print(a)
# dmax = 15
# dtest = np.array(range(dmax + 1)) 
# print(dtest)

# n = 10
# beta = np.array([1, 2, -1])
# x = np.linspace(0, 1, n).reshape(-1, 1)

# A = np.zeros((n, 2))
# A[:, 0] += 1
# A[:, 1] += x[:, 0]

# y = 1 + x + x ** 2
# estimated_beta = np.linalg.inv(A.T @ A) @ A.T @ y
# print(estimated_beta)
# x = np.linspace(0, 3, 100)
# estimated_y = estimated_beta[0] + estimated_beta[1] * x
# true_y = 1 + x + x ** 2
# plt.plot(x, estimated_y)
# plt.plot(x, true_y)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(['Estimated y', 'True y'])
# plt.show()
# u = np.array([[1], [2], [3], [4]])
# y = np.array([[10], [20], [11], [40]])
# from sklearn.model_selection import train_test_split

# def generate_U(u, d): 
#     n = len(u)
#     Xdly = np.zeros((n, d))
#     row = Xdly.shape[0]
#     Xdly_row = np.zeros((d, 1))
#     for i in range(row): 
#         for j in range(d): 
#             Xdly_row[j, :] = np.exp(-j * u[i, :] / d)
#         Xdly[i, :] = Xdly_row.reshape(1, -1)
#     return Xdly

# dmax = 11
# utr, uts, ytr, yts = train_test_split(u, y, test_size=0.5)
# dtest = np.arange(1, dmax)
# nd = len(dtest)
# mses = np.zeros(nd)

# for it, d in enumerate(dtest):
#     Utr = generate_U(utr, d)
#     Uts = generate_U(uts, d)

#     regr = LinearRegression().fit(Utr, ytr)
#     yhat = regr.predict(Uts)
#     mses[it] = np.mean((yhat - yts) ** 2)
# optimal_arg = np.argmin(mses)
# optimal_d = dtest[optimal_arg]
# print(optimal_d)

x5 = 0x00000000AAAAAAAA
x6 = 0x1234567812345678
x7 = x5 >> 16
x7 = x7 - 128
x7 = x7 >> 2
x7 = x7 & x6
# x7 = x6 << 4
# x7 = x5 >> 3
# x7 = x7 & 0xFEF
print('%x'%x7)
# max = 0b01111111111111111111111111111111
# min = -max - 1
# print(min - 128)
# print('{}'.format(max - 128))