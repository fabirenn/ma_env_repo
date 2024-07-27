import os
import sys

import keras.metrics
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet_model import unet
from metrics_calculation import pixel_accuracy, precision, mean_iou, dice_coefficient, recall, f1_score
from custom_callbacks import (
    ValidationCallback,
    clear_directory,
)
from data_loader import create_datasets_for_unet_training
from loss_functions import dice_loss

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

LOG_VAL_PRED = "data/predictions/unet"
CHECKPOINT_PATH = "artifacts/models/unet/unet_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512

EPOCHS = 100
PATIENCE = 30

os.environ["WANDB_DIR"] = "wandb/train_unet"
os.environ["WANDB_DATA_DIR"] = "/work/fi263pnye-ma_data/tmp"


def objective(trial):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8, 16])
    IMG_CHANNEL = trial.suggest_categorical("img_channel", [3, 8])
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.1)
    NUM_BLOCKS = trial.suggest_int("num_blocks", 3, 6)

    # Define the possible values for the number of filters
    filter_options = [16, 32, 64, 128, 256, 512]

    # Select the appropriate number of filters from the filter_options based on num_blocks
    filters_list = filter_options[:NUM_BLOCKS]

    try:
        train_dataset, val_dataset = create_datasets_for_unet_training(
            directory_train_images=TRAIN_IMG_PATH,
            directory_train_masks=TRAIN_MASK_PATH,
            directory_val_images=VAL_IMG_PATH,
            directory_val_masks=VAL_MASK_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            batch_size=BATCH_SIZE,
            channel_size=IMG_CHANNEL,
        )

        wandb.init(
            project="unet",
            entity="fabio-renn",
            mode="offline",
            name=f"train-unet-{trial.number}",
            config={
                "metric": "accuracy",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
            },
            dir=os.environ["WANDB_DIR"],
        )

        model = unet(
            IMG_WIDTH,
            IMG_HEIGHT,
            IMG_CHANNEL,
            DROPOUT_RATE,
            filters_list,
            training=True,
        )

        model.compile(
            optimizer="adam",
            loss=dice_loss,
            metrics=[
                "accuracy",
                pixel_accuracy,
                precision,
                mean_iou,
                dice_coefficient,
                f1_score,
                recall
            ],
        )

        history = model.fit(
            train_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=[
                WandbMetricsLogger(log_freq="epoch"),
                keras.callbacks.ModelCheckpoint(
                    filepath=CHECKPOINT_PATH,
                    save_best_only=True,
                    save_weights_only=False,
                    monitor="val_loss",
                    verbose=1,
                ),
                ValidationCallback(
                    model=model,
                    validation_data=val_dataset,
                    log_dir=LOG_VAL_PRED,
                    apply_crf=False,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=PATIENCE,
                    restore_best_weights=True,
                ),
            ],
        )

        val_loss = min(history.history["val_loss"])
        wandb.finish()

        return val_loss
    except tf.errors.ResourceExhaustedError:
        print(
            "Resource exhausted error caught. GPU may not have enough memory."
        )
        return float("inf")
    except Exception as e:
        print(f"An exception occurred: {e}")
        return float("inf")


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    # clear_directory("/work/fi263pnye-ma_data/tmp/artifacts")

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
