import numpy as np














def loss_eval(dL_dy, y, yhat, a, u): 
    u_sum = np.sum(u)
    au_sum = a * u_sum
    a_sum_u_sum = np.sum(a * u_sum)
    dy_du = (a_sum_u_sum - au_sum) / u_sum ** 2
    dL_du = dL_dy * dy_du
    return dL_du