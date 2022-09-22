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