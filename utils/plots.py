import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from typing import Sequence

# Make a grid from a batch of images
def make_grid(images,
              padding: int = 10,
              grid_sh: Sequence[int] = (2, 10),
              color: Sequence[int] = (255, 255, 255)):

    rows, cols = grid_sh
    color = np.array(color, dtype=np.uint8)
    imgs_h = []
    for row in range(rows):
        imgs_w = []
        for col in range(cols):
            idx = row * cols + col
            img = images[idx]
            sh = np.array(img.shape) + 2 * padding
            sh[-1] = img.shape[-1]
            new_img = np.ones(sh, dtype=np.uint8) * color[None, None, :]
            new_img[padding: sh[0] - padding, padding:sh[1] - padding, :] = img
            imgs_w.append(new_img)
        new_img = np.concatenate(imgs_w, axis=1)
        imgs_h.append(new_img)
    return np.concatenate(imgs_h)


# plot a batch of image into a grid
def plot_grid_images(imgs,
                     grid_shape: Sequence[int]):
    n_rows, n_cols = grid_shape
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    plt.title('Generated Images')
    for row in range(n_rows):
        for col in range(n_cols):
            index = row * n_cols + col
            plt.subplot(n_rows, n_cols, index + 1)
            img = imgs[index]
            plt.imshow(img)
            plt.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()


def make_save_gif(imgs,
                  name: str = 'test',
                  time: int = 250):
    imgs = [Image.fromarray(im) for im in imgs]
    imgs[0].save(name + ".gif", format="GIF", append_images=imgs,
                 save_all=True, duration=time, loop=0)
