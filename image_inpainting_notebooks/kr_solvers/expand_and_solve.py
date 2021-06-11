import numpy as np
from scipy.linalg import solve
import time
from copy import deepcopy
import matplotlib.pyplot as plt


def fill_image(K, net, corrupted_img, mask):
    _, w, h = corrupted_img.shape

    s = 0  ## Amount of downsampling
    for idx, arg in enumerate(net.args):
        if arg.name == 'downsample':
            s += 1

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

    print("Loaded and Expanded Kernel; Now Subsetting Observed Entries")
    obs_pairs = np.argwhere(mask[0, :,:] != 0)    
    p = len(obs_pairs)
    K_train = np.zeros((p,p), dtype='float32')

    for n1, pair1 in enumerate(obs_pairs):
        r1, c1 = pair1
        K_train[n1, :]  = K[r1, c1, obs_pairs[:, 0], obs_pairs[:, 1]]    
    
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
    print(y_test.shape)

    for c in range(imputed_img.shape[0]):
        imputed_img[c, unobs_pairs[:, 0], unobs_pairs[:, 1]] = y_test[c,:].reshape(-1)

    visualize_imputed_image(corrupted_img, imputed_img)
    

def visualize_imputed_image(corrupted_img, imputed_img):
    vis_corrupted = np.rollaxis(corrupted_img, 0, 3)
    imputed_img = np.clip(imputed_img, 0, 1)
    
    imputed_image = np.rollaxis(imputed_img, 0, 3)

    fig, ax = plt.subplots(1,2)
    ax[0].imshow(vis_corrupted)
    ax[0].axis("off")
    ax[1].imshow(imputed_image)
    ax[1].axis("off")    
    
