import numpy as np

def Jeval(w, a): 
    a1 = a[0]
    a2 = a[1]
    a3 = a[2]
    w1 = w[0]
    w2 = w[1]
    z1 = a1*w1*w2
    z2 = a2*w1+a3*w2**2
    J = z1*np.exp(z1*z2)

    dw1 = a1*w2*np.exp(z1*z2)+2*z1*a1*a2*w1*w2*np.exp(z1*z2)
    dw2 = a1*w1*np.exp(z1*z2)+z1*(a1*a2*w1**2+2*a1*a3*w2)*np.exp(z1*z2)
    Jgrad = np.array([dw1, dw2])

    return J, Jgrad