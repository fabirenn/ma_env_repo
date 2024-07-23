import os

import matplotlib.pyplot as plt
import numpy as np
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

        if test_images.ndim == 3 and test_images.shape[2] > 3:
            test_images = test_images[:, :, 3]

        plt.figure(figsize=(45, 15))

        plt.subplot(1, 3, 1)
        plt.title("GT")
        plt.imshow(test_images)

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(test_masks)

        plt.subplot(1, 3, 3)
        plt.title("Pred Mask")
        plt.imshow(predictions)

        file_name = f"val_pred_epoch{iterator+1}.png"
        plt.savefig(os.path.join(pred_img_path, file_name))
        plt.close()

    else:
        for i, testimage, prediction, testmask in zip(
            range, test_images, predictions, test_masks
        ):
            if testimage.ndim == 3 and testimage.shape[2] > 3:
                testimage = testimage[:, :, 3]
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
    Apply CRF to the prediction.

    Parameters:
    image: The original image
    prediction: The model's sigmoid output

    Returns:
    result: The CRF-refined segmentation
    """

    image = image.numpy() if isinstance(image, tf.Tensor) else image
    prediction = prediction.numpy() if isinstance(prediction, tf.Tensor) else prediction

    # Convert softmax output to unary potentials
    unary = unary_from_softmax(prediction.transpose((2, 0, 1)))
    unary = np.ascontiguousarray(unary)

    # Create the dense CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], prediction.shape[-1])
    d.setUnaryEnergy(unary)

    # Create pairwise potentials (bilateral and spatial)
    pairwise_gaussian = create_pairwise_gaussian(sdims=(3, 3), shape=image.shape[:2])
    d.addPairwiseEnergy(pairwise_gaussian, compat=1)

    pairwise_bilateral = create_pairwise_bilateral(sdims=(10, 10), schan=(5, 5, 5), img=image, chdim=2)
    d.addPairwiseEnergy(pairwise_bilateral, compat=1)

    # Perform inference
    Q = d.inference(2)

    # Convert the Q array to the final prediction
    result = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return result[..., np.newaxis]
