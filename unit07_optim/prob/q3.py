import numpy as np

def Jeval(a, b, x, y):
    z = np.sum(a*np.exp(-((x-b)**2)/2))
    J = np.sum(np.log(1+np.exp(z))-y*z)

    Jgrada = np.sum(((z+np.exp(z))/(1+np.exp(z))-y)*np.exp(-((x-b)**2)/2))
    Jgradb = np.sum(((z+np.exp(z))/(1+np.exp(z))-y)*a*(x-b)*np.exp(-((x-b)**2)/2))

    return J, Jgrada, Jgradb

a = np.arange(10).reshape(5, 2)
i = a.shape[0]
b = np.ones((i, 1))
a = np.hstack((b, a))
print(a)