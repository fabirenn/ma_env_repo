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

ORIGINAL_IMAGES_PATH = "data/backgrounds/background_new/"
FENCE_IMAGES_PATH = "data/greenscreen/greenscreen_new/"
FENCE_IMAGES_PATH_SEGMENTED = "data/new/segmented/"
FENCE_MASKS_PATH = "data/labels_new/"
DESTINATION_PATH = "data/new/generated/"

# RGB values for the relevant classes
CLASS_COLORS = {
    "wire": (51, 221, 255),
    "post": (241, 177, 195),
    "tensioner": (245, 147, 49),
}

background_images = []
fence_images = []
fence_masks = []

def extract_relevant_mask(mask, relevant_classes):
    """
    Extract a binary mask for the relevant classes from the multi-class mask.
    """
    binary_mask = np.zeros(mask.shape[:2], dtype=np.uint8)

    for class_name, rgb_value in relevant_classes.items():
        # Determine the channel index for this class
        channel_index = list(relevant_classes.keys()).index(class_name)

        # Debug: Verify if the channel contains non-zero values
        print(f"Channel {channel_index} unique values: {np.unique(mask[:, :, channel_index])}")

        # Create a binary mask for this class
        class_mask = mask[:, :, channel_index] > 0  # Non-zero values indicate class presence
        binary_mask[class_mask] = 1

    return binary_mask

def apply_mask(image, mask, class_value):
    class_mask = mask == class_value
    return np.where(class_mask[:, :, np.newaxis], image, 0)


def replace_fence_pixels_using_mask(background_image, fence_image, mask):
    """
    Replace black pixels in the mask with the background pixels,
    and other pixels with the fence structure.
    """
    #result_image = np.zeros_like(fence_image) 

    # Background pixels (where mask == 0)
    background_pixels = mask[..., 0] == 0  # Only check the first channel of the mask
    background_pixels = np.repeat(background_pixels[:, :, np.newaxis], 3, axis=2)

    # Fence pixels (where mask > 0)
    fence_pixels = ~background_pixels

    # Replace black pixels (background) with the background image
    #
    #background_image[background_pixels] = background_image[background_pixels]

    # Replace non-black pixels (fence) with the fence image
    fence_image[~background_pixels] = background_image[~background_pixels]

    return fence_image


def save_image(image, path):
    image = convert_to_uint8(image)
    Image.fromarray(image).save(str(path))


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


def visualize_mask(mask, title=""):
    h, w, c = mask.shape
    combined_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(1, c):  # Skip the background channel (0)
        class_mask = mask[:, :, i]
        color = (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
        combined_mask[class_mask == 1] = color
    plt.imshow(combined_mask)
    plt.title(title)
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
fence_masks = load_masks_from_directory(FENCE_MASKS_PATH)
fence_masks = resize_images(fence_masks, 3000, 2000)


for i in range(len(background_images)):
    print(f"Processing image {i + 1}")
    print(f"Background image shape: {background_images[i].shape}")
    print(f"Fence image shape: {fence_images[i].shape}")
    print(f"Mask shape: {fence_masks[i].shape}")

    background_images[i] = cv2.cvtColor(background_images[i], cv2.COLOR_BGR2RGB)
    fence_images[i] = cv2.cvtColor(fence_images[i], cv2.COLOR_BGR2RGB)


    # Replace pixels based on mask
    result_image = replace_fence_pixels_using_mask(
        background_images[i], fence_images[i], fence_masks[i]
    )
    
    # Save the generated image
    save_image(result_image, DESTINATION_PATH + f"image_{i + 1}.png")

'''   
for i in range(14):
    img = cv2_greenscreenremover(fence_images[i], i)
# make_binary_mask(img, i)


fence_images_new = load_images_from_directory(FENCE_IMAGES_PATH_SEGMENTED)
fence_images_new = resize_images(fence_images_new, 3000, 2000)
fence_masks = load_masks_from_directory(FENCE_MASKS_PATH)
fence_masks = resize_images(fence_masks, 3000, 2000)
# fence_masks = make_binary_masks(fence_masks, 30)


for i in range(14):
    print(f"Processing image {i + 100}")
    print(f"Background image shape: {background_images[i].shape}")
    print(f"Fence image shape: {fence_images_new[i].shape}")
    # print(f"Mask shape: {fence_masks[i].shape}")

    # visualize_mask(fence_masks[i], title=f"Mask {i + 100}")

    result_image = replace_fence_pixels(
        background_images[i], fence_images_new[i], fence_masks[i]
    )
    # show_image(result_image)
    save_image()
    save_image(result_image, DESTINATION_PATH + "image" + str(i + 100) + ".png")''' 
