import os
import numpy as np
from keras.models import load_model
from data_loader import create_dataset_for_mask_prediction
from processing import safe_predictions_locally
from segnet.custom_layers import custom_objects

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
    patches = create_patches(image, patch_size, overlap)
    segmented_patches = []

    for x, y, patch in patches:
        patch = np.expand_dims(patch, axis=0)
        prediction = model.predict(patch, batch_size=1)
        segmented_patches.append((x, y, prediction[0]))

    return reconstruct_image(segmented_patches, img_height, img_width, patch_size)

def create_patches(image, patch_size, overlap):
    patches = []
    img_height, img_width, _ = image.shape
    step_size = patch_size - overlap
    for y in range(0, img_height, step_size):
        for x in range(0, img_width, step_size):
            y_end = min(y + patch_size, img_height)
            x_end = min(x + patch_size, img_width)
            patch = np.zeros((patch_size, patch_size, image.shape[2]), dtype=image.dtype)
            patch[: y_end - y, : x_end - x, :] = image[y:y_end, x:x_end, :]
            patches.append((x, y, patch))
    return patches

def reconstruct_image(patches, img_height, img_width, patch_size):
    num_classes = patches[0][2].shape[-1]
    reconstructed_image = np.zeros((img_height, img_width, num_classes), dtype=np.float32)
    patch_count = np.zeros((img_height, img_width, num_classes), dtype=np.float32)
    for x, y, patch in patches:
        y_end = min(y + patch_size, img_height)
        x_end = min(x + patch_size, img_width)
        reconstructed_image[y:y_end, x:x_end, :] += patch[: y_end - y, : x_end - x, :]
        patch_count[y:y_end, x:x_end, :] += 1
    reconstructed_image /= np.maximum(patch_count, 1)
    return np.argmax(reconstructed_image, axis=-1).astype(np.int32)

# Environment setup
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
IMG_PATH = "data/originals/images"
model_paths = [
    "artifacts/models/unet/unet_checkpoint.keras",
    "artifacts/models/segnet/segnet_checkpoint.keras",
    "artifacts/models/deeplab/deeplab_checkpoint.keras",
    "artifacts/models/segan/segan_checkpoint.keras",
    "artifacts/models/ynet/ynet_checkpoint.keras",
]
model_names = [
    "unet",
    "segnet",
    "deeplab",
    "segan",
    "ynet"
    ]

# Load data
original_images, preprocessed_images = create_dataset_for_mask_prediction(img_dir=IMG_PATH)
print("Loaded images")

# Load and predict for each model
for model_path, model_name in zip(model_paths, model_names):
    print(f"Using Model: {model_name}")
    model_path_abs = os.path.abspath(model_path)
    if model_name == "segnet":
        model = load_model(model_path_abs, custom_objects=custom_objects, compile=False)
    else:
        model = load_model(model_path_abs, compile=False)

    for i, (original_image, preprocessed_image) in enumerate(zip(original_images, preprocessed_images)):
        if model_name == "unet" or model_name == "segan":
            segmented_image = segment_image(preprocessed_image, model, patch_size=512, overlap=50)
        else:
            segmented_image = segment_image(original_image, model, patch_size=512, overlap=50)
        
        safe_predictions_locally(
            range=None,
            iterator=i,
            test_images=original_image,
            predictions=segmented_image,
            test_masks=None,
            pred_img_path=f"data/predictions/{model_name}",
            val=False,
        )

    print(f"Saved predictions for model {model_name} in data/predictions/{model_name}")
