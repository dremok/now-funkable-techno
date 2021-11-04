import os
import random

import matplotlib.cm as cm
import tensorflow as tf
import tensorflow_hub as hub

from definitions import ROOT_DIR

# Load compressed models from tensorflow_hub
os.environ['TFHUB_MODEL_LOAD_FORMAT'] = 'COMPRESSED'

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['figure.figsize'] = (12, 12)
mpl.rcParams['axes.grid'] = False

import numpy as np
import PIL.Image

# Parameters
x_res, y_res = 1000, 1000
xmin, xmax = -1.5, 1.5
width = xmax - xmin
ymin, ymax = -1.5, 1.5
height = ymax - ymin

z_abs_max = 10
max_iter = 1000


def julia_set(c):
    # Initialise an empty array (corresponding to pixels)
    julia = np.zeros((x_res, y_res))

    # Loop over each pixel
    for ix in range(x_res):
        for iy in range(y_res):
            # Map pixel position to a point in the complex plane
            z = complex(ix / x_res * width + xmin,
                        iy / y_res * height + ymin)
            # Iterate
            iteration = 0
            while abs(z) <= z_abs_max and iteration < max_iter:
                z = z ** 2 + c
                iteration += 1
            iteration_ratio = iteration / max_iter
            # Set the pixel value to be equal to the iteration_ratio
            julia[ix, iy] = iteration_ratio

    # Plot the array using matplotlib's imshow
    fig, ax = plt.subplots()
    cmap = mpl.colors.ListedColormap(np.random.rand(256, 3))
    ax.imshow(julia, interpolation='nearest', cmap=cmap)
    plt.axis('off')
    plt.show()
    fig.savefig(f'{ROOT_DIR}/julia/images/content.png', dpi=750)


def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def load_img(path_to_img):
    max_dim = 1024
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img


def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)


if __name__ == "__main__":
    for i in range(6, 7):
        julia_set(complex(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)))

        content_image = load_img('/Users/maxleander/code/nft/julia/images/content.png')
        style_image = load_img(f'/Users/maxleander/code/nft/julia/images/{i}.jpeg')
        style_image = tf.image.resize(style_image, (256, 256))

        hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
        stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
        image = tensor_to_image(stylized_image)
        image.save(f'{ROOT_DIR}/julia/sample{i}.png')
