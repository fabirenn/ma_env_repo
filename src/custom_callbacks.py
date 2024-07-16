import os

import tensorflow as tf
import shutil
from keras import backend as K
from keras.callbacks import Callback
from keras.utils import array_to_img

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
        # print(f"Validation sample batch shape: {x_batch.shape}, {y_true_batch.shape}")
        y_pred_batch = self.model.predict(x_batch, verbose=1)
        # print(f"Prediction batch shape: {y_pred_batch.shape}")
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
        log_images_wandb(
            epoch=epoch,
            x=x_batch[0],
            y_true=y_true_batch[0],
            y_pred=y_pred_batch[0],
        )


def log_images_wandb(epoch, x, y_true, y_pred):
    columns = ["epoch", "original", "true_mask", "predicted_mask"]
    extract_first_three_channels = x[:, :, :3]
    input_image = wandb.Image(array_to_img(extract_first_three_channels))
    true_mask = wandb.Image(array_to_img(y_true))
    predicted_mask = wandb.Image(array_to_img(y_pred))
    wandb_table = wandb.Table(columns=columns)
    wandb_table.add_data(epoch, input_image, true_mask, predicted_mask)

    wandb.log({"val_image": wandb_table})


def dice_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + K.epsilon()) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + K.epsilon()
    )


def specificity_score(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    true_negatives = tf.reduce_sum(
        tf.cast(
            tf.logical_and(tf.logical_not(y_true), tf.logical_not(y_pred)),
            tf.float32,
        )
    )
    false_positives = tf.reduce_sum(
        tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), tf.float32)
    )

    specificity = true_negatives / (
        true_negatives + false_positives + K.epsilon()
    )
    return specificity

def clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)
