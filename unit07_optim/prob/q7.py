import numpy as np

def Jeval(P, x, y):
    z = x.T@P@x
    J = np.sum(z/y-np.log(z))

    JgradP = np.sum(1/y-1/z)*(x.T@x)

    return J, JgradP

def Jeval_with_loop(P, x, y):
    z = x.T@P@x
    n = y.shape[0]

    J = np.zeros(y.shape)
    for i in range(n): 
        J[i, :] = np.sum(z[i, :]/y[i, :]-np.log(z[i, :]))
    
    JgradP = np.zeros(y.shape)
    for i in range(n): 
        JgradP = np.sum(1/y[i, :]-1/z[i, :])*(x[i, :].T@x[i, :])

    return J, JgradP

a = np.arange(10).reshape(5, 2)
i = a.shape[0]
b = np.ones((i, 1))
a = np.hstack((b, a))
print(a)