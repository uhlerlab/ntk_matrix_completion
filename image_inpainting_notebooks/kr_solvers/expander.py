import numpy as np
import torch

def expand_kernel(K, s, w, h):

    d1 = K.shape[0]
    p = 2**s
    
    K_new = np.zeros((p, p, w, h), dtype='float32')
    for i in range(p):
        for j in range(p):
            C11 = K[i,j,:, :]
            x = p - i
            y = p - j
            C11 = np.roll(C11, x, axis=0)
            C11 = np.roll(C11, y, axis=1)
            C = np.ones((w, h)) * np.min(C11)
            C[:d1, :d1] = C11
            C = np.roll(C, -x, axis=0)
            C = np.roll(C, -y, axis=1)
            K_new[i, j, :, :] = C    
    
    K = np.zeros((w, h, w, h), dtype='float32')
    for i in range(w):
        for j in range(h):
            K[i,j,:,:] = np.roll(K_new[i%p,j%p,:,:], i - i%p, axis=0)
            K[i,j,:,:] = np.roll(K[i,j,:,:], j - j%p, axis=1)

    return K


def expand_eigenpro(K, s, w, h):
    
    d1 = K.shape[0]
    p = 2**s
    K_new = np.zeros((p, p, w, h), dtype='float32')
    for i in range(p):
        for j in range(p):
            C11 = K[i,j,:, :]
            x = p - i
            y = p - j
            C11 = np.roll(C11, x, axis=0)
            C11 = np.roll(C11, y, axis=1)
            C = np.ones((w, h)) * np.min(C11)
            C[:d1, :d1] = C11
            C = np.roll(C, -x, axis=0)
            C = np.roll(C, -y, axis=1)
            K_new[i, j, :, :] = C    

    return torch.from_numpy(K_new)
