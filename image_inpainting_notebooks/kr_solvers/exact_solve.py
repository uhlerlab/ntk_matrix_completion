import numpy as np
from scipy.linalg import solve
import time
from copy import deepcopy


def kr_solve(K, corrupted_img, mask, epsilon=0):


    print("Loaded Kernel; Now Subsetting Observed Entries")
    obs_pairs = np.argwhere(mask[0, :,:] != 0)
    p = len(obs_pairs)
    K_train = np.zeros((p,p), dtype='float32')
    
    for n1, pair1 in enumerate(obs_pairs):
        r1, c1 = pair1
        K_train[n1, :]  = K[r1, c1, obs_pairs[:, 0], obs_pairs[:, 1]]    
    K_train += np.eye(p) * np.trace(K_train)/p * epsilon
    
    unobs_pairs = np.argwhere(mask[0, :,:] == 0)
    K_test = np.zeros((p, len(unobs_pairs)), dtype='float32')
    for n1, pair1 in enumerate(obs_pairs):
        r1, c1 = pair1
        K_test[n1, :] = K[r1, c1, unobs_pairs[:, 0], unobs_pairs[:, 1]]    

    channels, rows, cols = np.nonzero(mask)
    y_obs = corrupted_img[channels, rows, cols]
    ys = []

    start = 0
    for c in range(mask.shape[0]):
        ys.append(y_obs[start:start+p].reshape(-1, 1))
        start += p
    ys = np.concatenate(ys, axis=-1)
    print("Starting exact solve for kernel regression")
    print("This system has " + str(K_train.shape[0]) + " equations & unknowns.")
    s = time.time()
    yK_inv = solve(K_train, ys, assume_a='sym').T
    print(time.time() - s, " time for solve")
    
    imputed_img = deepcopy(corrupted_img)
    y_test = yK_inv @ K_test

    for c in range(imputed_img.shape[0]):
        imputed_img[c, unobs_pairs[:, 0], unobs_pairs[:, 1]] = y_test[c,:].reshape(-1)

    return imputed_img
