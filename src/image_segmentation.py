import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image

from data_loader import create_dataset_for_image_segmentation
from processing import safe_predictions_locally


def create_patches(image, patch_size, overlap):
    """
    Split the image into overlapping patches.

    Parameters:
    image: The input image
    patch_size: The size of each patch
    overlap: The overlap between patches

    Returns:
    patches: A list of tuples (x, y, patch)
    """
    patches = []
    img_height, img_width, _ = image.shape
    step_size = patch_size - overlap

    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            y_end = min(y + patch_size, img_height)
            x_end = min(x + patch_size, img_width)
            patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
            patch[:y_end - y, :x_end - x, :] = image[y:y_end, x:x_end, :]
            patches.append((x, y, patch))

    return patches


def reconstruct_image(patches, img_height, img_width, patch_size):
    """
    Reconstruct the image from overlapping patches.

    Parameters:
    patches: A list of tuples (x, y, patch) with the segmented patches
    img_height: Height of the original image
    img_width: Width of the original image
    patch_size: The size of each patch
    overlap: The overlap between patches

    Returns:
    reconstructed_image: The reconstructed image
    """
    reconstructed_image = np.zeros((img_height, img_width), dtype=np.float32)
    patch_count = np.zeros((img_height, img_width), dtype=np.float32)

    for x, y, patch in patches:
        y_end = min(y + patch_size, img_height)
        x_end = min(x + patch_size, img_width)
        reconstructed_image[y:y_end, x:x_end] += patch[:y_end - y, :x_end - x]
        patch_count[y:y_end, x:x_end] += 1

    return reconstructed_image / patch_count


def segment_image(image, model, patch_size, overlap):
    """
    Segment the image using patches and reconstruct the full image.

    Parameters:
    image: The input image to segment
    model: The trained segmentation model
    patch_size: The size of each patch
    overlap: The overlap between patches

    Returns:
    segmented_image: The segmented full image
    """
    img_height, img_width, _ = image.shape
    patches = create_patches(
        image, patch_size, overlap
    )  # Split image into patches
    segmented_patches = []

    for x, y, patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, batch_size=1)  # Predict segmentation for the patch
        segmented_patch = prediction[
            0, :, :, 0
        ]  # Remove batch dimension and channel dimension
        binary_patch = (segmented_patch > 0.5).astype(np.float32)
        segmented_patches.append((x, y, binary_patch))

    # Reconstruct the full image from the segmented patches
    segmented_image = reconstruct_image(
        segmented_patches, img_height, img_width, patch_size
    )
    return segmented_image


CHECKPOINT_UNET = "artifacts/models/unet/unet_checkpoint.h5"
CHECKPOINT_SEGNET = "artifacts/models/segnet/segnet_checkpoint.h5"
CHECKPOINT_DEEPLAB = "artifacts/models/deeplab/deeplab_checkpoint.h5"
CHECKPOINT_SEGAN = "artifacts/models/segan/segan_checkpoint.h5"
CHECKPOINT_YNET = "artifacts/models/ynet/ynet_checkpoint.h5"

IMG_PATH = "data/generated"
MASK_PATH = "data/segmented/mask"

unet_model = load_model(CHECKPOINT_UNET, compile=False)
unet_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

original_images = []
original_masks = []
preprocessed_images = []
original_images, preprocessed_images, original_masks = create_dataset_for_image_segmentation(
    img_dir=IMG_PATH, mask_dir=MASK_PATH, unet=True
)
print("loaded the images")

for original_image, preprocessed_image, original_mask, i in zip(
    original_images, preprocessed_images, original_masks, range(10)
):
    segmented_image = segment_image(
        preprocessed_image, unet_model, patch_size=512, overlap=100
    )
    safe_predictions_locally(
        range=None,
        iterator=i,
        test_images=original_image,
        predictions=segmented_image,
        test_masks=original_mask,
        pred_img_path="data/predictions/originals",
        val=True
    )
