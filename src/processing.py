import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pydensecrf.densecrf as dcrf
import tensorflow as tf
from keras.utils import array_to_img, img_to_array
from pydensecrf.utils import (
    create_pairwise_bilateral,
    create_pairwise_gaussian,
    unary_from_softmax,
)

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


def apply_crf(image, prediction):
    """
    Apply CRF to the prediction.

    Parameters:
    image: The original image
    prediction: The model's sigmoid output

    Returns:
    result: The CRF-refined segmentation
    """

    image = image.numpy() if isinstance(image, tf.Tensor) else image
    prediction = (
        prediction.numpy() if isinstance(prediction, tf.Tensor) else prediction
    )

    # Convert sigmoid output to softmax for binary classification
    softmax = np.zeros((2, *prediction.shape))
    softmax[1, :, :] = prediction
    softmax[0, :, :] = 1 - prediction

    # Convert softmax output to unary potentials
    unary = unary_from_softmax(softmax)
    unary = np.ascontiguousarray(unary)

    # Create the dense CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], 2)
    d.setUnaryEnergy(unary)

    # Create pairwise potentials (bilateral and spatial)
    pairwise_gaussian = create_pairwise_gaussian(
        sdims=(3, 3), shape=image.shape[:2]
    )
    d.addPairwiseEnergy(pairwise_gaussian, compat=3)

    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(50, 50), schan=(20, 20, 20), img=image, chdim=2
    )
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Perform inference
    Q = d.inference(5)

    # Convert the Q array to the final prediction
    result = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return result[..., np.newaxis]
