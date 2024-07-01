import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.utils import array_to_img, img_to_array
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.spatial.distance import directed_hausdorff
from PIL import Image

import wandb
from processing import apply_crf_to_pred, safe_predictions_locally


class ValidationCallback(Callback):
    def __init__(self, model, validation_data, log_dir, apply_crf):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.apply_crf = apply_crf
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_sample = self.validation_data.take(1)
        x_batch, y_true_batch = next(iter(random_sample))
        y_pred_batch = self.model.predict(x_batch)
        if self.apply_crf is True:
            y_pred_batch[0] = apply_crf_to_pred(x_batch[0], y_pred_batch[0])

        safe_predictions_locally(
            range=None,
            iterator=epoch,
            test_images=x_batch[0],
            test_masks=y_true_batch[0],
            predictions=y_pred_batch[0],
            pred_img_path=self.log_dir,
            val=True,
        )
        # self.log_images_wandb(epoch, x_batch[0], y_true_batch[0],
        # y_pred_batch[0])

        logs = logs or {}
        
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        
        iou = iou_score(val_targ, val_predict)
        dice = dice_score(val_targ, val_predict)
        precision = precision_score(val_targ.flatten(), val_predict.flatten())
        recall = recall_score(val_targ.flatten(), val_predict.flatten())
        accuracy = accuracy_score(val_targ.flatten(), val_predict.flatten())
        specificity = specificity_score(val_targ, val_predict)
        hausdorff = hausdorff_distance(val_targ, val_predict)

        # Log metrics to Wandb
        wandb.log({
            "epoch": epoch + 1,
            "val_iou": iou,
            "val_dice": dice,
            "val_precision": precision,
            "val_recall": recall,
            "val_accuracy": accuracy,
            "val_specificity": specificity,
            "val_hausdorff": hausdorff
        })
        
        print(f"Epoch {epoch + 1} - val_iou: {iou} - val_dice: {dice} - val_precision: {precision} - val_recall: {recall} - val_accuracy: {accuracy} - val_specificity: {specificity} - val_hausdorff: {hausdorff}")

    def log_images_wandb(self, epoch, x, y_true, y_pred):
        columns = ["epoch", "original", "true_mask", "predicted_mask"]

        input_image = wandb.Image(array_to_img(x))
        true_mask = wandb.Image(array_to_img(y_true))
        predicted_mask = wandb.Image(array_to_img(y_pred))
        wandb_table = wandb.Table(columns=columns)
        wandb_table.add_data(epoch, input_image, true_mask, predicted_mask)

        wandb.log({"val_image": wandb_table})


def iou_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    return np.sum(intersection) / np.sum(union)


def dice_score(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    return 2 * np.sum(intersection) / (np.sum(y_true) + np.sum(y_pred))


def specificity_score(y_true, y_pred):
    true_negatives = np.logical_and(~y_true, ~y_pred)
    false_positives = np.logical_and(~y_true, y_pred)
    return np.sum(true_negatives) / (np.sum(true_negatives) + np.sum(false_positives))


def hausdorff_distance(y_true, y_pred):
    y_true_points = np.argwhere(y_true)
    y_pred_points = np.argwhere(y_pred)
    return max(directed_hausdorff(y_true_points, y_pred_points)[0], directed_hausdorff(y_pred_points, y_true_points)[0])
