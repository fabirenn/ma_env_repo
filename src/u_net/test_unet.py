import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import array_to_img, img_to_array
from PIL import Image as im
from unet_architecture_hcp import unet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_callbacks import ValidationCallback
from data_loader import (
    convert_to_tensor,
    create_dataset,
    load_images_from_directory,
    load_masks_from_directory,
    make_binary_masks,
    normalize_image_data,
    preprocess_images,
    resize_images,
)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TEST_IMG_PATH = "data/training_test/images_mixed"
TEST_MASK_PATH = "data/training_test/labels_mixed"
CHECKPOINT_PATH = "artifacts/models/unet/unet_checkpoint.h5"
PRED_IMG_PATH = "artifacts/models/unet/pred"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

BATCH_SIZE = 4
EPOCHS = 50


def calculate_binary_iou(pred_mask, true_mask):
    pred_mask = np.round(pred_mask).astype(
        int
    )  # Thresholding predictions to 0 or 1
    true_mask = true_mask.astype(int)

    intersection = np.logical_and(pred_mask, true_mask).sum()
    union = np.logical_or(pred_mask, true_mask).sum()

    if union == 0:
        return float("nan")  # Avoid division by zero
    else:
        return intersection / union


def calculate_binary_dice(pred_mask, true_mask):
    pred_mask = np.round(pred_mask).astype(
        int
    )  # Thresholding predictions to 0 or 1
    true_mask = true_mask.astype(int)

    intersection = 2 * np.logical_and(pred_mask, true_mask).sum()
    total = pred_mask.sum() + true_mask.sum()

    if total == 0:
        return float("nan")  # Avoid division by zero
    else:
        return intersection / total


def safe_predictions(range, test_images, predictions, test_masks):

    for i, testimage, prediction, testmask in zip(
        range, test_images, predictions, test_masks
    ):

        plt.figure(figsize=(45, 15))

        plt.subplot(1, 3, 1)
        plt.title("GT")
        plt.imshow(testimage)

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(testmask)

        plt.subplot(1, 3, 3)
        plt.title("Pred Mask")
        plt.imshow(prediction)

        file_name = f"pred_figure_{i}.png"
        plt.savefig(os.path.join(PRED_IMG_PATH, file_name))
        plt.close()


def add_prediction_to_list(test_dataset):
    predictions_list = []

    for image, mask in test_dataset:
        prediction = model.predict(image)
        for j in range(BATCH_SIZE):
            prediction_image = prediction[j]
            prediction_image = array_to_img(prediction_image)
            predictions_list.append(prediction_image)

    return predictions_list


# loading images and masks from their corresponding paths into to separate lists
test_images = load_images_from_directory(TEST_IMG_PATH)
test_masks = load_masks_from_directory(TEST_MASK_PATH)
print("Test-Images successfully loaded..")

# resizing the images to dest size for training
test_images = resize_images(test_images, IMG_WIDTH, IMG_HEIGHT)
test_masks = resize_images(test_masks, IMG_WIDTH, IMG_HEIGHT)
print("Test-Images resized")

# normalizing the values of the images and binarizing the image masks
test_images = normalize_image_data(test_images)
print("Test-Images normalized..")
test_images_preprocessed = preprocess_images(test_images)
print("Test-Images preprocessed..")
test_masks_binary = make_binary_masks(test_masks, 30)
print("Test-Masks binarized..")

# converting the images/masks to tensors + expanding the masks tensor slide to
# 1 dimension
#print(len(test_masks))
tensor_test_images = convert_to_tensor(test_images_preprocessed)
tensor_test_masks = convert_to_tensor(test_masks_binary)
tensor_test_masks = tf.expand_dims(tensor_test_masks, axis=-1)

print("Test images converted to tensors..")

test_dataset = create_dataset(
    tensor_test_images,
    tensor_test_masks,
    batchsize=4,
    buffersize=tf.data.AUTOTUNE,
)

model = load_model(CHECKPOINT_PATH, compile=False)
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

predictions = add_prediction_to_list(test_dataset)


# Calculate metrics for each image
ious = [
    calculate_binary_iou(pred, true)
    for pred, true in zip(predictions, test_masks_binary)
]
dices = [
    calculate_binary_dice(pred, true)
    for pred, true in zip(predictions, test_masks_binary)
]

# Average metrics over the dataset
mean_iou = np.nanmean(ious)
mean_dice = np.nanmean(dices)

print(f"Mean IoU: {mean_iou}")
print(f"Mean Dice Coefficient: {mean_dice}")

safe_predictions(
    range=range(212),
    test_images=test_images,
    predictions=predictions,
    test_masks=test_masks,
)
