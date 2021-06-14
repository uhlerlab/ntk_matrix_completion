import numpy as np
from scipy.linalg import solve
import time
from copy import deepcopy
import kr_solvers.expander as expander
import torch
import kr_solvers.eigenpro as eigenpro


USE_CUDA = torch.cuda.is_available()

def solve(K, corrupted_img, mask, max_iter=11):
    K /= np.max(K)
    K = torch.from_numpy(K)
    if USE_CUDA:
        K = K.cuda()
    imputed_img = compressed_kernel_solve(K, corrupted_img, mask,
                                          max_iter=max_iter+1)
    return imputed_img    

def expand_and_solve(K, corrupted_img, mask, expand_factor, max_iter=11):

    _, w, h = corrupted_img.shape

    K /= np.max(K)
    K = expander.expand_eigenpro(K, expand_factor, w, h)
    if USE_CUDA:
        K = K.cuda()
    imputed_img = compressed_kernel_solve(K, corrupted_img, mask,
                                          s=2**expand_factor,
                                          max_iter=max_iter+1)
    return imputed_img


def split_into_batches(X, bs):
    chunk = len(X) // bs
    chunks = []
    start = 0

    for t in range(chunk):
        if t < chunk - 1:
            end = start + bs
        else:
            end = len(X)
        chunks.append(X[start:end, :])
        start += bs
    if chunk == 0:
        chunks.append(X)
    return chunks


def compressed_kernel_solve(K, masked, mask,  max_iter=10, s=None):

    if s is not None:
        def kernel_fn(pair1, pair2):
            n1, _ = pair1.size()
            n2, _ = pair2.size()
            xr = pair1[:, 0].T
            yr = pair2[:, 0].view(-1,1)
            xr = xr.repeat(n2, 1).T
            yr = yr.repeat(1, n1).T
            
            xc = pair1[:, 1].T
            yc = pair2[:, 1].view(-1,1)
            xc = xc.repeat(n2, 1).T
            yc = yc.repeat(1, n1).T
            
            yr = yr + xr%s - xr
            yc = yc + xc%s - xc
            
            out = K[xr.long()%s, xc.long()%s,
                    yr.long(), yc.long()]
            del xc, yc, xr, yr
            return out / 2
    else:
        def kernel_fn(pair1, pair2):
            
            n1, _ = pair1.size()
            n2, _ = pair2.size()
            xr = pair1[:, 0].T
            yr = pair2[:, 0].view(-1,1)
            xr = xr.repeat(n2, 1).T
            yr = yr.repeat(1, n1).T
            
            xc = pair1[:, 1].T
            yc = pair2[:, 1].view(-1,1)
            xc = xc.repeat(n2, 1).T
            yc = yc.repeat(1, n1).T
            
            out = K[xr.long(), xc.long(),
                    yr.long(), yc.long()]
            del xc, yc, xr, yr
            return out / 2
        
    obs_idxs = np.argwhere(mask[0, :, :] != 0).astype('float32')
    p = len(obs_idxs)
    channels, rows, cols = np.nonzero(mask)
    y_obs = masked[channels, rows, cols]
    x_train = obs_idxs
    
    ys = []
    start = 0
    for c in range(mask.shape[0]):
        ys.append(y_obs[start:start+p].reshape(-1, 1))
        start += p
    y_train = np.concatenate(ys, axis=-1).astype('float32')
    
    n_class = mask.shape[0]  # Number of color channels
    Mem = 4
    epochs = list(range(0, max_iter, 10))
    device = torch.device("cuda" if USE_CUDA else "cpu")
    masked_entries = np.argwhere(mask[0, :, :] == 0).astype('float32')    
    x_test = torch.from_numpy(masked_entries)

    bs = 256
    chunks = split_into_batches(x_test, bs)

    print(x_train.shape, y_train.shape)
    print("Training Model")
    model = eigenpro.FKR_EigenPro(kernel_fn, x_train, n_class, device=device)
    metrics = model.fit(x_train, y_train, epochs=epochs, mem_gb=Mem)
    
    all_out = []
    for c in chunks:
        with torch.no_grad():
            if USE_CUDA:
                out = model.forward(c.cuda()).cpu().data.numpy()
            else:
                out = model.forward(c).data.numpy()
            c = c.cpu()
            all_out.append(out)
    out = np.concatenate(all_out, axis=0)
    imputed_img = deepcopy(masked)

    for idx, pair1 in enumerate(masked_entries):
        r1, c1 = pair1
        r1 = int(r1)
        c1 = int(c1)
        imputed_img[:, r1, c1] = out[idx]
    return imputed_img    
