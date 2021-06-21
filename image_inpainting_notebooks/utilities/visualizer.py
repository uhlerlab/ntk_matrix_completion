import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.figsize'] = [10, 5]


def visualize_images(*imgs):
    fig, ax = plt.subplots(1, len(imgs))
    
    for idx, img in enumerate(imgs):
        img = np.clip(img, 0, 1)
        if img.shape[0] == 1:
            img = img[0]
            ax[idx].imshow(1-img, cmap='Greys')
        else:
            img = np.rollaxis(img, 0, 3)
            ax[idx].imshow(img)
        ax[idx].show()
        
            
def visualize_kernel_slice(K, coordinate):
    w, h, w, h = K.shape
    x = np.array(list(range(w)))
    y = np.array(list(range(h)))
    xx, yy = np.meshgrid(x, y)
    r, c = coordinate

    new_slice = K[r, c, :, :]
    plt.contourf(xx, yy, new_slice)
    plt.colorbar()
    plt.ylim(max(plt.ylim()), min(plt.ylim()))
    plt.title('Kernel at Coordinate: (' + str(r) +  ',' + str(c) + ')')
    plt.show()
