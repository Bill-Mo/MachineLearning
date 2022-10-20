import numpy as np

def fit_linear(t, zt):
    x = -t
    y = zt

    xm = np.mean(x)
    ym = np.mean(y)
    syx = np.mean((y - ym) * (x - xm))
    sxx = np.mean((x - xm) ** 2)
    beta1 = syx/sxx
    beta0 = ym - beta1 * xm
    yhat = beta0 + beta1 * x
    RSS = np.sum((y - yhat) ** 2)

    alpha = beta1
    lnz0 = beta0
    z0 = np.exp(lnz0)
    return alpha, z0, RSS

def px(x):
    prev_fail_p = 1 
    for i in range(1, x):
        curr_success_p = 0.2 + 0.05 * (i - 1)
        prev_fail_p *= 1- curr_success_p
    curr_success_p = 0.2 + 0.05 * (x - 1)
    return prev_fail_p * curr_success_p

x =  np.linspace(1, 17, 17, dtype=int)
y = 0
for i in x:
    y += i * px(i)
print(y)