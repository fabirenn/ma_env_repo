import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from data_loader import (
    load_images_from_directory,
    load_masks_from_directory,
    make_binary_masks,
    resize_images,
)

ORIGINAL_IMAGES_PATH = "data/backgrounds/"
FENCE_IMAGES_PATH = "data/greenscreen/"
FENCE_IMAGES_PATH_SEGMENTED = "data/segmented/fence/"
FENCE_MASKS_PATH = "data/segmented/mask/"
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
    """Konvertiert ein Bild zu uint8, wobei das Bild normalisiert wird,
    falls nötig."""
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


def cv2_greenscreenremover(image, iterator):
    cv2.imwrite("image.png", image)
    img = cv2.imread("image.png")
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    a_channel = lab[:, :, 1]
    th = cv2.threshold(
        a_channel, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    masked = cv2.bitwise_and(img, img, mask=th)  # contains dark background
    m1 = masked.copy()
    m1[th == 0] = (255, 255, 255)
    mlab = cv2.cvtColor(masked, cv2.COLOR_BGR2LAB)
    dst = cv2.normalize(
        mlab[:, :, 1],
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )
    threshold_value = 100
    dst_th = cv2.threshold(dst, threshold_value, 255, cv2.THRESH_BINARY_INV)[1]
    mlab[:, :, 1][dst_th == 255] = 127
    img2 = cv2.cvtColor(mlab, cv2.COLOR_LAB2BGR)
    # correction_matrix = cv2.addWeighted(img2, 1.0, np.zeros_like(img2), 0,
    # -40)
    # img2 = cv2.addWeighted(img2, 0.8, correction_matrix, 0.2, 0)
    img2[th == 0] = (255, 255, 255)
    cv2.imwrite(
        FENCE_IMAGES_PATH_SEGMENTED + "image" + str(iterator + 100) + ".png",
        img2,
    )
    return img2


def make_binary_mask(img, iterator):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    # binary_mask = (binary_mask // 255).astype(np.uint8)

    cv2.imwrite(
        FENCE_MASKS_PATH + "mask" + str(iterator + 100) + ".png", binary_mask
    )


background_images = load_images_from_directory(ORIGINAL_IMAGES_PATH)
background_images = resize_images(background_images, 3000, 2000)
fence_images = load_images_from_directory(FENCE_IMAGES_PATH)
fence_images = resize_images(fence_images, 3000, 2000)

for i in range(len(background_images)):
    img = cv2_greenscreenremover(fence_images[i], i)
    make_binary_mask(img, i)


fence_images_new = load_images_from_directory(FENCE_IMAGES_PATH_SEGMENTED)
fence_masks = load_masks_from_directory(FENCE_MASKS_PATH)
fence_masks = resize_images(fence_masks, 3000, 2000)
fence_masks = make_binary_masks(fence_masks, 30)


for i in range(len(background_images)):
    result_image = replace_fence_pixels(
        background_images[i], fence_images_new[i], fence_masks[i]
    )
    # show_image(result_image)
    save_image(result_image, DESTINATION_PATH + "image" + str(i + 100) + ".png")
