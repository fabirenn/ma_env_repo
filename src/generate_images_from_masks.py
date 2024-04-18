import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

from data_loader import (
    load_images_from_directory,
    load_masks_from_directory,
    make_binary_masks,
    resize_images,
)

ORIGINAL_IMAGES_PATH = "data/vorrübergehend/original/"
FENCE_IMAGES_PATH = "data/vorrübergehend/greenscreen/"
FENCE_MASKS_PATH = "data/vorrübergehend/maske/"
DESTINATION_PATH = "data/generated/"

background_images = []
fence_images = []
fence_masks = []


def apply_mask(image, mask):
    return image * mask[:, :, np.newaxis]


def replace_fence_pixels(original_image, replacement_image, mask):
    masked_original = apply_mask(replacement_image, mask)
    # show_image(masked_original)

    masked_replacement = apply_mask(original_image, 1 - mask)
    # show_image(masked_replacement)

    result_image = masked_original + masked_replacement
    # show_image(result_image)
    return result_image


def save_image(image, path):
    image = convert_to_uint8(image)
    Image.fromarray(image).save(path)


def convert_to_uint8(image):
    """Konvertiert ein Bild zu uint8, wobei das Bild normalisiert wird, falls nötig."""
    if image.dtype != np.uint8:
        # Normalisierung nur, wenn die maximale Helligkeit über 255 liegt
        # if image.max() > 255:
        #    image = 255 * (image.astype(np.float64) / image.max())
        image = image.astype(np.uint8)
    return image


def show_image(image):
    plt.imshow(image)
    plt.axis("off")
    plt.show()


background_images = load_images_from_directory(ORIGINAL_IMAGES_PATH)
background_images = resize_images(background_images, 1440, 960)
fence_images = load_images_from_directory(FENCE_IMAGES_PATH)
fence_masks = load_masks_from_directory(FENCE_MASKS_PATH)
fence_masks = make_binary_masks(fence_masks, 30)

for i in range(len(background_images)):
    result_image = replace_fence_pixels(
        background_images[i], fence_images[i], fence_masks[i]
    )
    # show_image(result_image)
    save_image(result_image, DESTINATION_PATH + "image" + str(i) + ".png")
