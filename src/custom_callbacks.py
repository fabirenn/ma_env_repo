import os
import shutil

import keras
import numpy as np
import tensorflow as tf
from keras.utils import array_to_img

import wandb
from processing import apply_crf_to_pred, safe_predictions_locally

class_colors = {
    0: (0, 0, 0),  # Background (Black)
    1: (51, 221, 255),
    2: (241, 177, 195),
    3: (245, 147, 49),
    4: (102, 255, 102),
}


def map_class_to_color(mask):
    """Map each class in the mask to its corresponding color."""
    color_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in class_colors.items():
        color_mask[mask == class_id] = color
    return color_mask


class ValidationCallback(keras.callbacks.Callback):
    def __init__(self, model, validation_data, log_dir, apply_crf, log_wandb):
        super().__init__()
        self.validation_data = validation_data
        self.log_dir = log_dir
        self.model = model
        self.apply_crf = apply_crf
        self.log_wandb = log_wandb
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_sample = self.validation_data.take(1)
        x_batch, y_true_batch = next(iter(random_sample))
        # print(f"Validation sample batch shape: {x_batch.shape}, {y_true_batch.shape}")
        x = x_batch[0]
        y_true = y_true_batch[0]

        x_rgb = x[..., :3][..., ::-1]

        try:
            y_pred = self.model.predict(tf.expand_dims(x, axis=0), verbose=1)[0]
            if self.apply_crf:
                y_pred = apply_crf_to_pred(x, y_pred)

            y_pred_class = np.argmax(y_pred, axis=-1)
            y_pred_colored = map_class_to_color(y_pred_class)

            y_true_class = np.argmax(y_true, axis=-1)
            y_true_colored = map_class_to_color(y_true_class)

            safe_predictions_locally(
                range=None,
                iterator=epoch,
                test_images=x_rgb,
                test_masks=y_true_colored,
                predictions=y_pred_colored,
                pred_img_path=self.log_dir,
                val=True,
            )
            if self.log_wandb:
                log_images_wandb(
                    epoch=epoch,
                    x=x_rgb,
                    y_true=y_true_colored,
                    y_pred=y_pred_colored,
                )
        except Exception as e:
            print(f"Error during prediction in on_epoch_end: {e}")


def log_images_wandb(epoch, x, y_true, y_pred):
    columns = ["epoch", "original", "true_mask", "predicted_mask"]
    extract_first_three_channels = x[:, :, :3]
    input_image = wandb.Image(array_to_img(extract_first_three_channels))
    true_mask = wandb.Image(array_to_img(y_true))
    predicted_mask = wandb.Image(array_to_img(y_pred))
    wandb_table = wandb.Table(columns=columns)
    wandb_table.add_data(epoch, input_image, true_mask, predicted_mask)

    wandb.log({"val_image": wandb_table})


def clear_directory(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
    os.makedirs(directory_path)
