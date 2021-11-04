import os
import random
import shutil

import requests
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

CONTENT_FILE = f'{ROOT_DIR}/julia/images/content.png'

STYLE_IMG_URLS = [
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/composition-8-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/royal-red-and-blue-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/starry-night-the-famous-painting-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/beheading-of-saint-john-the-baptist-the-famous-pai-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/guernica-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/night-watch-the-famous-painting-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-persistence-of-memory-the-famous-painting-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/luncheon-on-the-boating-party-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/no-5-1948-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-lillies-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/landscape-with-full-of-icarus-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/las-meninas-the-famous-painting-4.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/three-musicians-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/a-sunday-afternoon-on-the-island-of-la-grande-jatt-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/impression-sunrise-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-triumph-of-galatea-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/view-of-toledo-1.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-tower-of-babel-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-dance-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-swing-1.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-flower-carrier-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-kiss-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/1200px-Vincent_Willem_van_Gogh_102.jpg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/whistlers-mother-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/bal-du-moulin-de-la-galette-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/magritte1icon.jpg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/cafe-terrace-at-night-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/american-gothic-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/les-demoiselles-davignon-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-grand-odalisque-1.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-liberty-leading-the-people-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/napoleon-crossing-the-alps-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-sleeping-gypsy-1.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/primavera-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/dogs-playing-poker-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/portrait-of-dora-maar-2.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/massacre-of-the-innocents-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/portrait-of-madame-recamier-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/sistine-chapel-ceiling-1.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-birth-of-venus-1.jpeg',
    'https://i0.wp.com/bookmypainting.com/wp-content/uploads/2019/06/1200px-Van_Eyck_-_Arnolfini_Portrait.jpg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/school-of-athens-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/olympia-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-third-of-may-a-famous-painting-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/creation-of-adam-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the-scream-famous-painting-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/the_last_supper-the-famous-painting-2.jpeg',
    'https://i2.wp.com/bookmypainting.com/wp-content/uploads/2019/06/girl-with-a-pearl-earring-the-famous-painting-2.jpeg',
    'https://i1.wp.com/bookmypainting.com/wp-content/uploads/2019/06/mona-lisa-the-famous-paintings-2.jpeg',
]

# Parameters
x_res, y_res = 1000, 1000
xmin, xmax = -1.5, 1.5
width = xmax - xmin
ymin, ymax = -1.5, 1.5
height = ymax - ymin

z_abs_max = 10
max_iter = 1000


def create_julia_fractal(c):
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
    # plt.show()
    fig.savefig(CONTENT_FILE, dpi=750)


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


def fetch_style_img(i):
    style_image_url = STYLE_IMG_URLS[i]
    style_image_filename = style_image_url.split("/")[-1]
    r = requests.get(style_image_url, stream=True)
    if r.status_code == 200:
        r.raw.decode_content = True
        with open(style_image_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)
        return style_image_filename
    else:
        return None


def main():
    create_julia_fractal(complex(random.uniform(-0.9, 0.9), random.uniform(-0.9, 0.9)))

    i = random.randint(0, len(STYLE_IMG_URLS))
    while True:
        style_image_filename = fetch_style_img(i)
        if style_image_filename:
            break

    style_image = load_img(style_image_filename)
    style_image = tf.image.resize(style_image, (256, 256))
    hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')
    content_image = load_img(CONTENT_FILE)
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    image = tensor_to_image(stylized_image)
    image.save(f'{ROOT_DIR}/julia/sample.png')


if __name__ == "__main__":
    main()
