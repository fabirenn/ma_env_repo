import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import array_to_img, img_to_array

import wandb


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


def safe_predictions_locally(
    range, iterator, test_images, predictions, test_masks, pred_img_path, val
):
    if val is True:
        plt.figure(figsize=(45, 15))

        plt.subplot(1, 3, 1)
        plt.title("GT")
        plt.imshow(test_images)

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(test_masks, cmap=plt.cm.gray)

        plt.subplot(1, 3, 3)
        plt.title("Pred Mask")
        plt.imshow(predictions, cmap=plt.cm.gray)

        file_name = f"val_pred_epoch{iterator+1}.png"
        plt.savefig(os.path.join(pred_img_path, file_name))
        plt.close()

    else:
        for i, testimage, prediction, testmask in zip(
            range, test_images, predictions, test_masks
        ):
            plt.figure(figsize=(45, 15))

            plt.subplot(1, 3, 1)
            plt.title("GT")
            plt.imshow(testimage)

            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(testmask, cmap=plt.cm.gray)

            plt.subplot(1, 3, 3)
            plt.title("Pred Mask")
            plt.imshow(prediction, cmap=plt.cm.gray)

            file_name = f"test_pred_{i+1}.png"
            plt.savefig(os.path.join(pred_img_path, file_name))
            plt.close()


def add_prediction_to_list(test_dataset, model, batch_size):
    predictions_list = []
    binary_predictions = []
    for image, mask in test_dataset:
        prediction = model.predict(image)
        for j in range(batch_size):
            prediction_image = prediction[j]
            binary_prediction_image = (prediction_image > 0.5).astype(np.uint8)
            binary_predictions.append(binary_prediction_image)
            prediction_image = array_to_img(prediction_image)
            predictions_list.append(prediction_image)

    return predictions_list, binary_predictions
