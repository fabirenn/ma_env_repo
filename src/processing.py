import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
import tensorflow as tf
from keras.utils import array_to_img
from pydensecrf.utils import (
    create_pairwise_bilateral,
    create_pairwise_gaussian,
    unary_from_softmax,
)

# Define the color mapping for each class in BGR order
class_colors = {
    0: (0, 0, 0),  # Background (Black)
    1: (51, 221, 255),
    2: (241, 177, 195),
    3: (245, 147, 49),
    4: (102, 255, 102),
}


def map_class_to_color(mask):
    """Map each class in the mask to its corresponding color."""
    color_mask = np.zeros((*mask.shape[:2], 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color
    return color_mask


def safe_predictions_locally(
    range, iterator, test_images, predictions, test_masks, pred_img_path, val
):
    if val is True:

        if test_images.ndim == 3 and test_images.shape[2] > 3:
            test_images_rgb = cv2.cvtColor(test_images, cv2.COLOR_BGR2RGB)

        if predictions.ndim == 3 and predictions.shape[2] > 3:
            predictions = np.argmax(predictions, axis=-1)
        predictions_colored = map_class_to_color(predictions)

        if test_masks.ndim == 3 and test_masks.shape[2] > 3:
            test_masks = np.argmax(test_masks, axis=-1)
        test_masks_colored = map_class_to_color(test_masks)

        plt.figure(figsize=(45, 15))

        plt.subplot(1, 3, 1)
        plt.title("GT")
        plt.imshow(test_images_rgb)

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(test_masks_colored)

        plt.subplot(1, 3, 3)
        plt.title("Pred Mask")
        plt.imshow(predictions_colored)

        file_name = f"val_pred_epoch{iterator+1}.png"
        plt.savefig(os.path.join(pred_img_path, file_name))
        plt.close()

    else:
        for i, testimage, prediction, testmask in zip(
            range, test_images, predictions, test_masks
        ):
            if testimage.ndim == 3 and testimage.shape[2] > 3:
                test_image_rgb = cv2.cvtColor(testimage, cv2.COLOR_BGR2RGB)

            if prediction.ndim == 3 and prediction.shape[2] > 3:
                prediction = np.argmax(prediction, axis=-1)
            predictions_colored = map_class_to_color(predictions)

            if testmask.ndim == 3 and testmask.shape[2] > 3:
                testmask = np.argmax(testmask, axis=-1)
            test_masks_colored = map_class_to_color(test_masks)

            plt.figure(figsize=(45, 15))

            plt.subplot(1, 3, 1)
            plt.title("GT")
            plt.imshow(test_image_rgb)

            plt.subplot(1, 3, 2)
            plt.title("True Mask")
            plt.imshow(test_masks_colored)

            plt.subplot(1, 3, 3)
            plt.title("Pred Mask")
            plt.imshow(predictions_colored)

            file_name = f"test_pred_{i+1}.png"
            plt.savefig(os.path.join(pred_img_path, file_name))
            plt.close()


def add_prediction_to_list(test_dataset, model, batch_size, apply_crf):
    predictions_list = []
    colored_predictions = []
    for image, mask in test_dataset:
        prediction = model.predict(image)
        for j in range(batch_size):
            prediction_image = prediction[j]
            if apply_crf is True:
                prediction_image = apply_crf_to_pred(
                    image=image[j], prediction=prediction_image
                )
            # Convert to class indices
            prediction_class = np.argmax(prediction_image, axis=-1)

            # Map to color
            prediction_colored = map_class_to_color(prediction_class)

            predictions_list.append(array_to_img(prediction_colored))
            colored_predictions.append(prediction_colored)

    return predictions_list, colored_predictions


def apply_crf_to_pred(image, prediction):
    """
    Apply CRF post-processing to the model outputs.
    :param image: Original image (H, W, 3).
    :param output_probs: Probability map output by the model (H, W, num_classes).
    :return: Refined segmentation map.
    """
    h, w = image.shape[:2]
    num_classes = prediction.shape[-1]

    # Create the dense CRF model
    d = dcrf.DenseCRF2D(w, h, num_classes)

    # Get unary potentials (negative log of probabilities)
    unary = unary_from_softmax(prediction.transpose(2, 0, 1))
    d.setUnaryEnergy(unary)

    # Create pairwise potentials
    gaussian_pairwise = create_pairwise_gaussian(sdims=(3, 3), shape=(w, h))
    d.addPairwiseEnergy(gaussian_pairwise, compat=3)

    bilateral_pairwise = create_pairwise_bilateral(sdims=(50, 50), schan=(13, 13, 13), img=image, chdim=2)
    d.addPairwiseEnergy(bilateral_pairwise, compat=10)
    
    # Perform inference
    Q = d.inference(5)
    return np.argmax(Q, axis=0).reshape((h, w))
