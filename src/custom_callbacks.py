import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.utils import array_to_img, img_to_array
from PIL import Image
from scipy.spatial.distance import directed_hausdorff
from sklearn.metrics import accuracy_score, precision_score, recall_score

import wandb
from processing import apply_crf_to_pred, safe_predictions_locally


class ValidationCallback(Callback):
    def __init__(self, model, train_data, validation_data, log_dir, apply_crf):
        super().__init__()
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.apply_crf = apply_crf
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_sample = self.validation_data.take(1)
        x_batch, y_true_batch = next(iter(random_sample))
        y_pred_batch = self.model.predict(x_batch, verbose=1)
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

        # Training data predictions and metrics
        all_train_predict, all_train_mask = [], []
        for batch in self.train_data:
            images, labels = batch
            train_predict = (np.asarray(self.model.predict(images, verbose=0))).round()

            all_train_predict.append(train_predict)
            all_train_mask.append(labels.numpy())

        train_predict = np.concatenate(all_train_predict, axis=0)
        train_targ = np.concatenate(all_train_mask, axis=0)

        train_iou = iou_score(train_targ, train_predict)
        train_dice = dice_score(train_targ, train_predict)
        train_precision = precision_score(
            train_targ.flatten(), train_predict.flatten()
        )
        train_recall = recall_score(
            train_targ.flatten(), train_predict.flatten()
        )
        train_accuracy = accuracy_score(
            train_targ.flatten(), train_predict.flatten()
        )
        train_specificity = specificity_score(train_targ, train_predict)
        train_hausdorff = hausdorff_distance(train_targ, train_predict)

        # Validation data predictions and metrics
        all_val_predict, all_val_targ = [], []
        
        for batch in self.validation_data:
            images, labels = batch
            val_predict = (np.asarray(self.model.predict(images, verbose=0))).round()
            
            all_val_predict.append(val_predict)
            all_val_targ.append(labels.numpy())
        
        val_predict = np.concatenate(all_val_predict, axis=0)
        val_targ = np.concatenate(all_val_targ, axis=0)

        val_iou = iou_score(val_targ, val_predict)
        val_dice = dice_score(val_targ, val_predict)
        val_precision = precision_score(
            val_targ.flatten(), val_predict.flatten()
        )
        val_recall = recall_score(val_targ.flatten(), val_predict.flatten())
        val_accuracy = accuracy_score(val_targ.flatten(), val_predict.flatten())
        val_specificity = specificity_score(val_targ, val_predict)
        val_hausdorff = hausdorff_distance(val_targ, val_predict)

        # Log metrics to Wandb
        wandb.log(
            {
                "epoch": epoch + 1,
                "train_iou": train_iou,
                "train_dice": train_dice,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_accuracy": train_accuracy,
                "train_specificity": train_specificity,
                "train_hausdorff": train_hausdorff,
                "val_iou": val_iou,
                "val_dice": val_dice,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_accuracy": val_accuracy,
                "val_specificity": val_specificity,
                "val_hausdorff": val_hausdorff,
            }
        )
        
        print(
            f"Epoch {epoch + 1} - train_iou: {train_iou} - train_dice: {train_dice} - train_precision: {train_precision} - train_recall: {train_recall} - train_accuracy: {train_accuracy} - train_specificity: {train_specificity} - train_hausdorff: {train_hausdorff}"
        )
        print(
            f"Epoch {epoch + 1} - val_iou: {val_iou} - val_dice: {val_dice} - val_precision: {val_precision} - val_recall: {val_recall} - val_accuracy: {val_accuracy} - val_specificity: {val_specificity} - val_hausdorff: {val_hausdorff}"
        )

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
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    
    true_negatives = np.logical_and(~y_true, ~y_pred)
    false_positives = np.logical_and(~y_true, y_pred)
    
    tn = np.sum(true_negatives)
    fp = np.sum(false_positives)
    
    specificity = tn / (tn + fp)
    return specificity


def hausdorff_distance(y_true, y_pred):
    y_true_points = np.argwhere(y_true)
    y_pred_points = np.argwhere(y_pred)
    return max(
        directed_hausdorff(y_true_points, y_pred_points)[0],
        directed_hausdorff(y_pred_points, y_true_points)[0],
    )
