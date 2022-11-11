import numpy as np

def Jeval(w, b, x, y): 
    yhat = x*w+b
    J = np.sum((np.log(y)-np.log(yhat))**2)

    Jgradw = np.sum(2*(np.log(y)-np.log(yhat))*x)
    Jgradb = np.sum(2*(np.log(y)-np.log(yhat)))

    return J, Jgradw, Jgradb

a = np.arange(20).reshape(10, 2)
b = np.array([3, 0])
w = 2
x = a*b+w
print(x)