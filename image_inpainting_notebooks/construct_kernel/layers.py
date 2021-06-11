import numpy as np
import math
from multiprocessing import Process, Queue
from tqdm import tqdm

NUM_THREADS = 1

class Sequential():

    def __init__(self, *args):
        # Args needs to be a list of Module instances
        self.args = args
        flag = True
        for arg in args:
            if not isinstance(arg, Module):
                flag = False
                break
        assert flag, f"Input to Sequential should be Modules"
        assert self.args[0].name == 'conv', f"First layer needs to be convolutional"

        nonlinearities = set(["ReLU", "LeakyReLU"])
        flag = True
        for idx, arg in enumerate(args):
            if arg.name == 'conv' and idx != len(args) - 1:
                if args[idx + 1].name not in nonlinearities:
                    flag = False
        assert flag, f"Please follow Conv() layer with either ReLU() or LeakyReLU()"
        
    def get_ntk(self, d, Xs=None, num_threads=1):
        global NUM_THREADS        
        NUM_THREADS = num_threads
        
        pairs = []
        for i in range(d):
            for j in range(d):
                pairs.append((i,j))
                
        if Xs is None:
            Sigma_0 = np.ones((d, d, d, d), dtype='float32') * 1/400
            for i in range(d):
                for j in range(d):
                    Sigma_0[i,j,i,j] = 1/300

        else:
            Sigma_0 = np.zeros((d, d, d, d), dtype='float32')
            count = 0
            tmp_pairs = np.array(pairs)
            for pair1 in pairs:
                count += 1
                r1, c1 = pair1
                for V in Xs:
                    Sigma_0[r1, c1, :, :] += V[r1, c1] * V

        #nonlinearity = self.args[1]
        #Sigma, dSigma, K = mp_update_sigma(Sigma_0, pairs, d,
        #                                   nonlinearity.kap_0,
        #                                   nonlinearity.kap_1,
        #                                   first=True,
        #                                   kernel_size=self.args[0].kernel_size)
        
        for idx in tqdm(range(0, len(self.args))):
            if idx == 0:
                nonlinearity = self.args[1]
                Sigma, dSigma, K = mp_update_sigma(Sigma_0, pairs, d,
                                                   nonlinearity.kap_0,
                                                   nonlinearity.kap_1,
                                                   first=True,
                                                   kernel_size=self.args[0].kernel_size)
                continue
            arg = self.args[idx]
            if arg.name == 'conv':
                if idx < len(self.args) - 1:
                    #print(self.args[idx].name, self.args[idx + 1].name)
                    nonlinearity = self.args[idx + 1]                    
                    Sigma, dSigma, K = arg.update(Sigma, dSigma, K, pairs,
                                                  nonlinearity.kap_0,
                                                  nonlinearity.kap_1)
                else:
                    K = arg.update(Sigma, dSigma, K, pairs, None, None, last=True)
            elif 'ReLU' in arg.name:
                pass
            else:
                Sigma, dSigma, K, pairs = arg.update(Sigma, dSigma, K)
        return K
                
class Module():

    def __init__():
        self.name = ''
    
class Conv(Module):

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        self.name = 'conv'

    def update(self, Sigma, dSigma, K, pairs,
               kap_0, kap_1, 
               last=False):
        d = Sigma.shape[0]

        K_ = mp_update_K(pairs, Sigma, dSigma, K, d,
                         kernel_size=self.kernel_size)
        if last:
            return K_

        Sigma_, dSigma_ = mp_update_sigma(Sigma, pairs, d,
                                          kap_0, kap_1,
                                          kernel_size=self.kernel_size)
        return Sigma_, dSigma_, K_        

    
class Downsample(Module):

    def __init__(self):
        self.name = 'downsample'

    def update(self, Sigma, dSigma, K):
        d = Sigma.shape[0]
        return downsample(Sigma, dSigma, K, d)
    
class Upsample(Module):

    def __init__(self, bilinear=False):
        self.name = 'upsample'
        self.bilinear = bilinear

    def update(self, Sigma, dSigma, K):
        d = Sigma.shape[0]
        return upsample(Sigma, dSigma, K, d, bilinear=self.bilinear)

    
# Dual Activation for ReLU
class ReLU(Module):

    def __init__(self):
        self.name = 'ReLU'
        
    def kap_0(self, x):
        x = min(x, 1)
        x = max(x, -1)
        return 1/math.pi * (math.pi - math.acos(x))

    def kap_1(self, x):
        x = min(x, 1)
        x = max(x, -1)
        return 1/math.pi * (x * (math.pi - math.acos(x)) + math.sqrt(1 - x**2))

# Dual Activation for LeakyReLU
class LeakyReLU(Module):

    def __init__(self):
        self.name = 'LeakyReLU'
        
    def kap_0(self, x):
        x = min(x, 1)
        x = max(x, -1)
        b = .01
        correction = 2*b/(math.pi * (b**2 + 1))
        correction *= math.acos(x)
        return 1/math.pi * (math.pi - math.acos(x))  + correction
    
    def kap_1(self, x):
        x = min(x, 1)
        x = max(x, -1)
        b = .01
        correction = 2*b/(math.pi * (b**2 + 1))
        correction *= (x*math.acos(x) - math.sqrt(1-x**2))
        return 1/math.pi * (x * (math.pi - math.acos(x)) + math.sqrt(1 - x**2)) + correction    

# Circular padding for a coordinate entry in a d dimensional image
def circ_pad(x, d):
    return x % d


# The update for K for coordinate pair (pair1, pair2) in the CNTK
def K_fn(pair1, pair2, Sigma, dSigma, prev_K, d, kernel_size=3):

    K = 0
    r1, c1 = pair1
    r2, c2 = pair2

    l = - (kernel_size - 1) // 2
    u = (kernel_size - 1) // 2 + 1

    for m in range(l, u):
        for n in range(l, u):
            m_r1 = circ_pad(r1 + m, d)
            m_r2 = circ_pad(r2 + m, d)
            n_c1 = circ_pad(c1 + n, d)
            n_c2 = circ_pad(c2 + n, d)

            Z = Sigma[m_r1, n_c1, m_r2, n_c2]
            dZ =  dSigma[m_r1, n_c1, m_r2, n_c2]
            dZ *= prev_K[m_r1, n_c1, m_r2, n_c2]
            K +=  Z +  dZ
    return K

# Update Sigma and \dot{Sigma} for the coordinate pair (pair1, pair2) in the CNTK
def update_Z_fn(Sigma_0, pair1, pair2, d, kap_0, kap_1, kernel_size=3):
    r1, c1 = pair1
    r2, c2 = pair2
    arg = 0
    norm1 = 0
    norm2 = 0
    l = - (kernel_size - 1) // 2
    u = (kernel_size - 1) // 2 + 1

    for m in range(l, u):
        for n in range(l, u):

            m_r1 = circ_pad(r1 + m, d)
            m_r2 = circ_pad(r2 + m, d)
            n_c1 = circ_pad(c1 + n, d)
            n_c2 = circ_pad(c2 + n, d)

            prev_norm1 = Sigma_0[m_r1, n_c1, m_r1, n_c1]
            norm1 += prev_norm1
            prev_norm2 = Sigma_0[m_r2, n_c2, m_r2, n_c2]
            norm2 += prev_norm2
            prev_arg = Sigma_0[m_r1, n_c1, m_r2, n_c2]
            arg += prev_arg

    if norm1 == 0 or norm2 == 0:
        return 0, 0, 0

    s = kap_1(arg/math.sqrt(norm1 * norm2)) * math.sqrt(norm1 * norm2)
    ds = kap_0(arg/math.sqrt(norm1 * norm2))
    k = arg

    # The following scaling is optional, but is used in Arora et al. for the CNTK.
    # It just helps make sure the Sigma, dSigma values do not grow too large.
    s *= 1 / (kernel_size**2)
    ds *= 1 / (kernel_size**2)

    return s, ds, k


#Update Sigma across a minibatch of coordinate pairs
def update_sigma(Sigma_0, start, end, all_pairs, d, q,
                 kap_0, kap_1, 
                 first=False,
                 kernel_size=3):
    for count, pair1 in enumerate(all_pairs[start:end]):
        #print(count, end - start, "Sigma Update")
        Sigma_new = np.zeros((d,d), dtype='float32')
        dSigma_new = np.zeros((d,d), dtype='float32')
        if first:
            K_new = np.zeros((d,d), dtype='float32')
        for pair2 in all_pairs:
            r1, c1 = pair1
            r2, c2 = pair2
            s, ds, k = update_Z_fn(Sigma_0,
                                   pair1,
                                   pair2,
                                   d,
                                   kap_0, kap_1, 
                                   kernel_size=kernel_size)
            Sigma_new[r2, c2] = s
            dSigma_new[r2, c2] = ds
            if first:
                K_new[r2, c2] = k
        if first:
            q.put((r1, c1, Sigma_new, dSigma_new, K_new))
        else:
            q.put((r1, c1, Sigma_new, dSigma_new))


# Multithread the CNTK computation
def mp_update_sigma(Sigma_0, all_pairs, d, kap_0, kap_1, first=False, kernel_size=3):

    processes = []
    p = len(all_pairs)
    chunk_size = p // NUM_THREADS
    start = 0
    q = Queue()

    for t in range(NUM_THREADS):
        if t < NUM_THREADS - 1:
            end = start + chunk_size
        else:
            end = p
        if end - start <= 0:
            continue

        pr = Process(target=update_sigma,
                     args=(Sigma_0, start, end, all_pairs,
                           d, q, kap_0, kap_1, first, kernel_size))
        start += chunk_size
        processes.append(pr)

    for idx, pr in enumerate(processes):
        pr.start()
    Sigma_new = np.zeros((d,d,d,d), dtype='float32')
    dSigma_new = np.zeros((d,d,d,d), dtype='float32')
    
    # The first update is special since we need to initialize K
    if first:
        K_new = np.zeros((d,d,d,d), dtype='float32')

    while 1:
        running = any(pr.is_alive() for pr in processes)
        while not q.empty():
            if first:
                r1, c1, S, dS, K = q.get(False, .01)
                K_new[r1, c1, :, :] = K
            else:
                r1, c1, S, dS = q.get(False, .01)
            Sigma_new[r1, c1, :, :] = S
            dSigma_new[r1, c1, :, :] = dS

        if not running:
            break

    if first:
        return Sigma_new, dSigma_new, K_new
    else:
        return Sigma_new, dSigma_new
        

# Update K for a minibatch of image coordinates
def update_K(all_pairs, Sigma_1, dSigma_1, K_0, start, end, d, q, kernel_size=3):
    for count, pair1 in enumerate(all_pairs[start:end]):
        #print(count, end - start)
        K_new = np.zeros((d, d), dtype='float32')
        for pair2 in all_pairs:
            r1, c1 = pair1
            r2, c2 = pair2
            K_new[r2, c2] = K_fn(pair1, pair2,
                                 Sigma_1, dSigma_1, K_0, d, kernel_size=kernel_size)
        q.put((r1, c1, K_new))
    

# Multithread the updates for kernel coordinates
def mp_update_K(all_pairs, Sigma_1, dSigma_1, K_0, d, kernel_size=3):

    processes = []
    p = len(all_pairs)
    chunk_size = p // NUM_THREADS
    start = 0
    q = Queue()

    for t in range(NUM_THREADS):
        if t < NUM_THREADS - 1:
            end = start + chunk_size
        else:
            end = p
        if end - start <= 0:
            continue

        pr = Process(target=update_K,
                     args=(all_pairs, Sigma_1, dSigma_1, K_0, start, end, d, q,
                           kernel_size))
        start += chunk_size
        processes.append(pr)

    for idx, pr in enumerate(processes):
        pr.start()

    K_new = np.zeros((d, d, d, d), dtype='float32')
    while 1:
        running = any(pr.is_alive() for pr in processes)
        while not q.empty():
            r1, c1, val = q.get(False, .01)
            K_new[r1, c1, :, : ] = val

        if not running:
            break
    return K_new


# Nearest Neighbor Downsampling operator for kernel
def downsample(Sigma, dSigma, K, d):
    s = d//2
    Sigma_ = np.zeros((s, s, s, s), dtype='float32')
    dSigma_ = np.zeros((s, s, s, s), dtype='float32')
    K_ = np.zeros((s, s, s, s), dtype='float32')

    all_pairs = []
    for i in range(d):
        for j in range(d):
            all_pairs.append((i,j))

    
    for pair1 in all_pairs:
        for pair2 in all_pairs:
            r1, c1 = pair1
            r2, c2 = pair2
            if r2 % 2 == 1 or c2 % 2 == 1:
                continue
            if r1 % 2 == 1 or c1 % 2 == 1:
                continue
            r1_ = r1//2
            c1_ = c1//2
            r2_ = r2//2
            c2_ = c2//2

            Sigma_[r1_, c1_, r2_, c2_] = Sigma[r1, c1, r2, c2]
            dSigma_[r1_, c1_, r2_, c2_] = dSigma[r1, c1, r2, c2]
            K_[r1_, c1_, r2_, c2_] = K[r1, c1, r2, c2]

    new_pairs = []
    for i in range(s):
        for j in range(s):
            new_pairs.append((i, j))
    return Sigma_, dSigma_, K_, new_pairs

# Helper for bilinear upsampling
def bilinear_upsample(d, p1):
    r1, c1 = p1

    K = (d-1)/ (2*d - 1)
    sr1, sc1 = r1 * K, c1 * K
    sr1 = min(int(sr1), d - 2)
    sc1 = min(int(sc1), d - 2)

    Q11 = [sr1, sc1]
    Q12 = [sr1, sc1 + 1]
    Q21 = [sr1 + 1, sc1]
    Q22 = [sr1 + 1, sc1 + 1]

    C = (2 * d -1) / (d - 1)

    lr, ur = C * sr1, C * (sr1 + 1)
    lc, uc = C * sc1, C * (sc1 + 1)

    X = np.array([ur - r1, r1 - lr])
    Y = np.array([uc - c1, c1 - lc])

    K = 1/ ((ur - lr) * (uc - lc))
    return [(Y[0]*X[0] * K, Q11),
            (Y[0]*X[1] * K, Q21),
            (Y[1]*X[0] * K, Q12),
            (Y[1]*X[1] * K, Q22)]



# Perform upsampling for a minibatch of image coordinates
# If bilinear flag is False, use nearest neighbor upsampling.
# Otherwise uses bilinear upsampling
def sub_upsample(new_pairs, coeffs, Sigma, dSigma, K, start, end, s, q, bilinear):

    for idx1, pair1 in enumerate(new_pairs[start:end]):
        #print("UPSAMPLING: ", idx1, end - start)
        Sigma1 = np.zeros((s, s), dtype='float32')
        dSigma1 = np.zeros((s, s), dtype='float32')
        K1 = np.zeros((s, s), dtype='float32')
        for idx2, pair2 in enumerate(new_pairs):
            if not bilinear:
                r1, c1 = pair1
                r2, c2 = pair2
                r1_ = r1 // 2
                r2_ = r2 // 2
                c1_ = c1 // 2
                c2_ = c2 // 2
                Sigma1[r2, c2] = Sigma[r1_, c1_, r2_, c2_]
                dSigma1[r2, c2] = dSigma[r1_, c1_, r2_, c2_]
                K1[r2, c2] = K[r1_, c1_, r2_, c2_]
            else:
                r1, c1 = pair1
                r2, c2 = pair2
                for ck1, pk1 in coeffs[idx1 + start]:
                    for ck2, pk2 in coeffs[idx2]:
                        Sigma1[r2, c2] += ck1 * ck2 * Sigma[pk1[0], pk1[1],
                                                            pk2[0], pk2[1]]
                        dSigma1[r2, c2] += ck1 * ck2 * dSigma[pk1[0],
                                                              pk1[1],
                                                              pk2[0],
                                                              pk2[1]]
                        K1[r2, c2] += ck1 * ck2 * K[pk1[0],
                                                    pk1[1],
                                                    pk2[0],
                                                    pk2[1]]
        q.put((r1, c1, Sigma1, dSigma1, K1))

# Multithread upsampling across image coordinates for a d dimensional image
def upsample(Sigma, dSigma, K, d, bilinear=False):
    s = d * 2
    new_pairs = []
    for i in range(s):
        for j in range(s):
            new_pairs.append((i,j))
    Sigma_ = np.zeros((s, s, s, s), dtype='float32')
    dSigma_ = np.zeros(Sigma_.shape, dtype='float32')
    K_ = np.zeros(Sigma_.shape, dtype='float32')

    coeffs = []
    if bilinear:
        for pair1 in new_pairs:
            coeffs.append(bilinear_upsample(d, pair1))

    processes = []
    p = len(new_pairs)
    chunk_size = p // NUM_THREADS
    start = 0
    q = Queue()

    for t in range(NUM_THREADS):
        if t < NUM_THREADS - 1:
            end = start + chunk_size
        else:
            end = p
        if end - start <= 0:
            continue
        pr = Process(target=sub_upsample,
                     args=(new_pairs, coeffs,
                           Sigma, dSigma, K, start, end, s, q, bilinear))

        start += chunk_size
        processes.append(pr)

    for idx, pr in enumerate(processes):
        pr.start()
    Sigma_ = np.zeros((s, s, s, s)).astype('float32')
    dSigma_ = np.zeros(Sigma_.shape).astype('float32')
    K_ = np.zeros(Sigma_.shape).astype('float32')
    
    while 1:
        running = any(pr.is_alive() for pr in processes)
        while not q.empty():
            r1, c1, s_Sig, s_dSig, s_K = q.get(False, .01)
            Sigma_[r1, c1, :, :] = s_Sig
            dSigma_[r1, c1, :, :] = s_dSig
            K_[r1, c1, :, : ] = s_K

        if not running:
            break
    return Sigma_, dSigma_, K_, new_pairs
        
