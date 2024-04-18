import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import Callback
from keras.utils import array_to_img

import wandb


class ValidationCallback(Callback):
    def __init__(self, model, validation_data, log_dir="logs/images"):
        super().__init__()
        self.model = model
        self.validation_data = validation_data
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        random_sample = self.validation_data.take(1)
        x_batch, y_true_batch = next(iter(random_sample))
        y_pred_batch = self.model.predict(x_batch)
        self.log_images(epoch, x_batch[0], y_true_batch[0], y_pred_batch[0])

    def log_images(self, epoch, x, y_true, y_pred):
        columns = ["epoch", "original", "true_mask", "predicted_mask"]

        input_image = wandb.Image(array_to_img(x))
        true_mask = wandb.Image(array_to_img(y_true))
        predicted_mask = wandb.Image(array_to_img(y_pred))
        wandb_table = wandb.Table(columns=columns)
        wandb_table.add_data(epoch, input_image, true_mask, predicted_mask)

        wandb.log({"val_image": wandb_table})

        plt.figure(figsize=(10, 3))
        plt.subplot(1, 3, 1)
        plt.title("Input Image")
        plt.imshow(x)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("True Mask")
        plt.imshow(np.squeeze(y_true), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(np.squeeze(y_pred), cmap="gray")
        plt.axis("off")

        # Save the plot to a file
        file_name = f"epoch_{epoch + 1}.png"
        plt.savefig(os.path.join(self.log_dir, file_name))
        plt.close()


# [optional] use wandb.config as your config
