import numpy as np

def transform(x):
    xmat = x[:, 0]
    for i in range(1, x.shape[0]): 
        xmat = np.concatenate((xmat, x[:, i]), axis=0)
    return xmat

x=np.array([
            [0, 0, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 1, 0], 
            [0, 0, 1, 0],
])
print(transform(x))