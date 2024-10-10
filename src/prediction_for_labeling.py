import os
import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from keras.models import load_model
from data_loader import create_dataset_for_mask_prediction
from loss_functions import dice_loss
from processing import safe_predictions_locally
from metrics_calculation import dice_coefficient, pixel_accuracy

# Set GPU allocator (if needed)
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

# Define the path to the U-Net model
unet_model_path = "artifacts/models/unet/unet_checkpoint.keras"

# Load the pre-trained U-Net model
unet_model = load_model(unet_model_path, compile=False)

# Compile the U-Net model with the dice loss and accuracy metric
unet_model.compile(
    loss=dice_loss,
    metrics=['accuracy']
)

# Define image and mask directories
IMG_PATH = "data/unseen/images"
MASK_PATH = "data/unseen/masks"

# Load dataset
original_images = create_dataset_for_mask_prediction(
    img_dir=IMG_PATH
)
print("Loaded images")


class_colors = {
    0: (0, 0, 0),  # Background (Black)
    1: (51, 221, 255),
    2: (241, 177, 195),
    3: (245, 147, 49),
    4: (102, 255, 102),
}


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
    # Initialize arrays to store class probabilities
    num_classes = patches[0][2].shape[-1]  # Assuming the number of classes from the first patch
    reconstructed_image = np.zeros((img_height, img_width, num_classes), dtype=np.float32)
    patch_count = np.zeros((img_height, img_width, num_classes), dtype=np.float32)

    for x, y, patch in patches:
        y_end = min(y + patch_size, img_height)
        x_end = min(x + patch_size, img_width)
        
        # Sum the class probabilities for overlapping patches
        reconstructed_image[y:y_end, x:x_end, :] += patch[: y_end - y, : x_end - x, :]
        patch_count[y:y_end, x:x_end, :] += 1
    
    # Avoid division by zero and normalize the patches
    reconstructed_image /= np.maximum(patch_count, 1)

    return np.argmax(reconstructed_image, axis=-1).astype(np.int32)


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
            patch = np.zeros(
                (patch_size, patch_size, image.shape[2]), dtype=image.dtype
            )
            patch[: y_end - y, : x_end - x, :] = image[y:y_end, x:x_end, :]
            patches.append((x, y, patch))

    return patches


# Function to segment the image using the U-Net model
def segment_image(image, model, patch_size=512, overlap=50):
    # Create patches from the image
    img_height, img_width, _ = image.shape
    patches = create_patches(image, patch_size, overlap)
    segmented_patches = []

    for x, y, patch in patches:
        patch = np.expand_dims(patch, axis=0)  # Add batch dimension
        prediction = model.predict(patch, batch_size=1)
        segmented_patches.append((x, y, prediction[0]))

    # Reconstruct the full segmented image from patches
    segmented_image = reconstruct_image(segmented_patches, img_height, img_width, patch_size)
    return segmented_image


def convert_mask_to_rgb(mask, class_colors):
    """
    Convert a mask with class labels to an RGB image using a predefined color map.

    Parameters:
    mask: A 2D array of class labels
    class_colors: A dictionary mapping class indices to RGB colors

    Returns:
    rgb_image: A 3D array representing the RGB image
    """
    height, width = mask.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_index, color in class_colors.items():
        rgb_image[mask == class_index] = color
    
    return rgb_image


# Iterate over the images and predict segmentation masks
for i, original_image in enumerate(original_images):
    
    print(f"Processing image {i+1}")
    
    # Segment the image using the U-Net model
    segmented_image = segment_image(original_image, unet_model, patch_size=256, overlap=10)

    # Convert segmented image (with class labels) to RGB
    segmented_image_rgb = convert_mask_to_rgb(segmented_image, class_colors)
    
    # Save the RGB segmented mask as a JPG file in the MASK_PATH
    output_mask_path = os.path.join(MASK_PATH, f"segmented_mask_{i+1}.jpg")
    
    # Use PIL to save the image as JPG
    segmented_image_pil = Image.fromarray(segmented_image_rgb)
    segmented_image_pil.save(output_mask_path, "JPEG")
    
    print(f"Saved segmented mask {i+1} as {output_mask_path}")

print("Segmentation complete. Predictions saved.")

