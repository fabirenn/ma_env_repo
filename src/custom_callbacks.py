import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.utils import array_to_img, img_to_array
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

    def log_images_wandb(self, epoch, x, y_true, y_pred):
        columns = ["epoch", "original", "true_mask", "predicted_mask"]

        input_image = wandb.Image(array_to_img(x))
        true_mask = wandb.Image(array_to_img(y_true))
        predicted_mask = wandb.Image(array_to_img(y_pred))
        wandb_table = wandb.Table(columns=columns)
        wandb_table.add_data(epoch, input_image, true_mask, predicted_mask)

        wandb.log({"val_image": wandb_table})

    def log_images_locally(self, epoch, x, y_true, y_pred):

        # original_image = array_to_img(x)
        # file_name = f"epoch_{epoch + 1}_original.png"
        # original_image.save(os.path.join(self.log_dir, file_name))

        # true_mask = array_to_img(y_true)
        # file_name = f"epoch_{epoch + 1}_truemask.png"
        # true_mask.save(os.path.join(self.log_dir, file_name))

        prediction = array_to_img(y_pred)
        file_name = f"epoch_{epoch + 1}_prediction.png"
        prediction.save(os.path.join(self.log_dir, file_name))
        plt.close()
