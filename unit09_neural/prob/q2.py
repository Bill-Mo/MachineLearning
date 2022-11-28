import numpy as np
import matplotlib.pyplot as plt

# a) 
# nx = 50
# x = np.linspace(-3, 1, nx)
# wh = np.array([-1, 1, 1])
# wh = wh[:, None]
# bh = np.array([-1, 1, -2])
# bh = bh[:, None]
# zh = wh * x + bh
# uh = np.maximum(zh, 0)

# for i in range(nx):
#     x_plt = [x[i]] * 3
#     y_plt = uh[:, i]
#     # print(x_plt)
#     # print(y_plt)
#     plt.plot(x_plt, y_plt)
# plt.show()


# c d) 
x = np.array([-2, -1, -0, 3, 3.5])
x = np.linspace(-5, 5, 100)
y = np.array([0, 0, 1, 3, 3])
wh = np.array([-1, 1, 1])
wh = wh[:, None]
bh = np.array([-1, 1, -2])
bh = bh[:, None]
wo = np.array([0, 1, -1])
wo = wo[:, None]
bo = 0

zh = wh * x + bh
uh = np.maximum(zh, 0)
zo = np.sum(wo * uh + bo, axis=0)
print(zo.shape)
yhat = zo

# c)
N = 5
dL_dw = -2 / N * np.sum(uh * (y - wo * uh - bo))
dL_db = -2 / N * np.sum(y - wo * uh - bo)

# d)
plt.plot(x, yhat)
plt.grid()
plt.show()


# e)
def predict(x): 
    wh = np.array([-1, 1, 1])
    wh = wh[:, None]
    bh = np.array([-1, 1, -2])
    bh = bh[:, None]
    wo = np.array([0, 1, -1])
    wo = wo[:, None]
    bo = 0

    zh = wh * x + bh
    uh = np.maximum(zh, 0)
    zo = np.sum(wo * uh + bo, axis=0)
    print(zo.shape)
    yhat = zo + zh

    return yhat