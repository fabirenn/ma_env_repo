import os

import keras
import numpy as np
from keras.models import load_model
import tensorflow as tf
import cv2
import wandb
from metrics_calculation import dice_coefficient, calculate_class_iou, pixel_accuracy
from data_loader import create_dataset_for_image_segmentation
from loss_functions import combined_loss, dice_loss, iou_loss
from processing import safe_predictions_locally, apply_crf_to_pred
from segnet.custom_layers import custom_objects


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


def segment_image(image, model, patch_size, overlap, apply_crf):
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
        prediction = model.predict(
            patch, batch_size=1
        )  # Predict segmentation for the patch
        if apply_crf:
            prediction = apply_crf_to_pred(patch, prediction)
        
        segmented_patches.append((x, y, prediction[0]))

    # Reconstruct the full image from the segmented patches
    segmented_image = reconstruct_image(
        segmented_patches, img_height, img_width, patch_size
    )
    return segmented_image


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

model_paths = [
    "artifacts/models/unet/unet_checkpoint.keras",
    #"artifacts/models/segnet/segnet_checkpoint.keras",
    "artifacts/models/deeplab/deeplab_checkpoint.keras",
    #"artifacts/models/segan/segan_checkpoint.keras",
    #"artifacts/models/ynet/ynet_checkpoint.keras",
]
model_names = [
    "unet",
   #"segnet",
   "deeplab",
    #"segan",
    #"ynet"
    ]

'''
IMG_PATH = "data/generated"
MASK_PATH = "data/segmented/mask"'''

IMG_PATH = "data/originals/images"
MASK_PATH = "data/originals/masks"

os.environ["WANDB_DIR"] = "wandb/testing_models"

wandb.init(
    project="image-segmentation",
    entity="fabio-renn",
    name="image-segmentation",
    mode="offline",
    config={"metric": "accuracy"},
    dir=os.environ["WANDB_DIR"],
)

wandb.config

original_images = []
original_masks = []
preprocessed_images = []
original_images, preprocessed_images, original_masks = (
    create_dataset_for_image_segmentation(img_dir=IMG_PATH, mask_dir=MASK_PATH)
)
print("Loaded the images")

for i, model_path, model_name in zip(range(6), model_paths, model_names):
    print("Testing Model: " + model_name)
    model_path_abs = os.path.abspath(model_path)

    log_data = {}

    if model_name == "segnet":
        print("segnet=true")
        model = load_model(
            model_path_abs, custom_objects=custom_objects, compile=False
        )
    else:
        model = load_model(model_path_abs, compile=False)

    model.compile(
        loss=dice_loss,
        metrics=[
            "accuracy",
        ],
    )

    metrics_log = {
        "model": model_name,
        "dice": [],
        "pixel_accuracy": [],
        "iou_class_0": [],
        "iou_class_1": [],
        "iou_class_2": [],
        "iou_class_3": [],
        "iou_class_4": [],
        
    }

    for original_image, preprocessed_image, original_mask, i in zip(
        original_images, preprocessed_images, original_masks, range(11)
    ):
        #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if model_name not in ("unet", "segan", "ynet"):
            # print("no preprocessed images")
            segmented_image = segment_image(
                original_image, model, patch_size=512, overlap=50, apply_crf=False
            )
        elif model_name == "deeplab":
            segmented_image = segment_image(
                original_image, model, patch_size=512, overlap=50, apply_crf=True
            )
        else:
            segmented_image = segment_image(
                original_image, model, patch_size=512, overlap=50, apply_crf=False
            )
        
        # Convert original_mask from one-hot encoding to class labels
        original_mask_labels = np.argmax(original_mask, axis=-1)                    

        if segmented_image.shape != original_mask_labels.shape:
            raise ValueError(f"Shape mismatch: Original mask has shape {original_mask_labels.shape} but segmented mask has shape {segmented_image.shape}")
        
        # Calculate metrics
        iou_per_class = []
        dice_per_class = []

        for class_index in range(5):
            iou = calculate_class_iou(original_mask_labels, segmented_image, class_index).numpy()
            if not np.isnan(iou):  # Avoid NaN values in the log
                metrics_log[f"iou_class_{class_index}"].append(iou)

        # Ensure both the original mask and the segmented image are in the same type before passing to the dice_coefficient
        original_mask_labels_float = tf.cast(original_mask_labels, tf.float32)
        segmented_image_float = tf.cast(segmented_image, tf.float32)
         
        dice = dice_coefficient(original_mask_labels_float, segmented_image_float).numpy()
        pixel_acc = pixel_accuracy(original_mask_labels, segmented_image).numpy()

        metrics_log["dice"].append(dice)
        metrics_log["pixel_accuracy"].append(pixel_acc)

        safe_predictions_locally(
            range=None,
            iterator=i,
            test_images=original_image,
            predictions=segmented_image,
            test_masks=original_mask,
            pred_img_path="data/predictions/" + model_name,
            val=True,
        )

    # log the calculated to wandb to the corresponding wandb
    # Calculate average metrics for the model
    # Log alle IoU-Werte dynamisch für die Anzahl der Klassen
    for class_index in range(5):
        iou_list = metrics_log[f"iou_class_{class_index}"]  # Get the list for each IoU class
        if iou_list:  # Check if the list is not empty before calculating the mean
            log_data[f"{model_name}_iou_class_{class_index}"] = np.mean(iou_list)
        else:
            log_data[f"{model_name}_iou_class_{class_index}"] = None

    # Logge auch die durchschnittlichen Dice- und Genauigkeitswerte
    log_data.update({
            f"{model_name}_average_dice": np.mean(metrics_log["dice"]),
            f"{model_name}_average_pixel_accuracy": np.mean(metrics_log["pixel_accuracy"]),
        })
    wandb.log(log_data)

    print("Saved predictions in data/predictions/" + model_name)

wandb.finish()
