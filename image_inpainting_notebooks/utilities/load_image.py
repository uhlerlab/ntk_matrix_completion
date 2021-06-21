from PIL import Image
import numpy as np
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]
    
    img_cropped = img.crop(bbox)
    return img_cropped


def load_image(fname, img_width=None, img_height=None):
    if img_width is None or img_height is None:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize((img_height, img_width)),
                                        transforms.ToTensor()])
    img = Image.open(fname)    
    img = crop_image(img, 64)
    img = transform(img)
    img = img.numpy()

    mask = np.random.randint(0, 2, size=img.shape)[0]

    new_mask = np.zeros(img.shape)
    for i in range(len(new_mask)):
        new_mask[i, :,: ] = mask
    mask = new_mask
    
    corrupted_img = mask * img
    
    return img, corrupted_img, mask
    
def visualize_corrupted_image(clean_img, corrupted_img):

    fig, ax = plt.subplots(1,2)
    if clean_img.shape[0] > 1:
        vis_corrupted = np.rollaxis(corrupted_img, 0, 3)
        clean_image = np.rollaxis(clean_img, 0, 3)
        ax[0].imshow(clean_image)
        ax[0].axis("off")
        ax[1].imshow(vis_corrupted)
        ax[1].axis("off")

    else:
        vis_corrupted = corrupted_img[0]
        clean_image = clean_img[0]
        cmap='Greys'
        ax[0].imshow(1-clean_image, cmap=cmap)
        ax[0].axis("off")    
        ax[1].imshow(1-vis_corrupted, cmap=cmap)
        ax[1].axis("off")
    ax[0].show()
    ax[1].show()
    
