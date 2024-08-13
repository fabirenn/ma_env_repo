import os
import sys

import keras.metrics
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ynet_model import build_ynet, build_feature_extractor_for_pretraining, build_ynet_with_pretrained_semantic_extractor

from custom_callbacks import ValidationCallback
from data_loader import create_datasets_for_unet_training
from loss_functions import dice_loss
from metrics_calculation import (
    dice_coefficient,
    mean_iou,
    pixel_accuracy,
    precision,
    recall,
)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

LOG_VAL_PRED = "data/predictions/ynet"
CHECKPOINT_PATH_PRETRAINED = "artifacts/models/ynet/ynet_checkpoint_pretrained.keras"
CHECKPOINT_PATH_YNET = "artifacts/models/ynet/ynet_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

EPOCHS = 100
PATIENCE = 30

os.environ["WANDB_DIR"] = "wandb/train_unet"
os.environ["WANDB_DATA_DIR"] = "/work/fi263pnye-ma_data/tmp"


def objective(trial):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_categorical(
        "batch_size", [8, 12, 16, 20, 24, 28, 32]
    )
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.1)

    # Set the optimizer parameters
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

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
            project="ynet",
            entity="fabio-renn",
            mode="offline",
            name=f"train-ynet-{trial.number}",
            config={
                "metric": "accuracy",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
            },
            dir=os.environ["WANDB_DIR"],
        )

        # create model & start training it
        semantic_extractor_model = build_feature_extractor_for_pretraining(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE)
        semantic_extractor_model.load_weights(CHECKPOINT_PATH_PRETRAINED)

        model = build_ynet_with_pretrained_semantic_extractor(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE, semantic_extractor_model)

        # Create the SGD optimizer
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            decay=weight_decay,
            nesterov=False  # You can set this to True if you want to use Nesterov momentum
        )

        model.compile(
            optimizer=optimizer,
            loss=dice_loss,
            metrics=[
                "accuracy",
                pixel_accuracy,
                precision,
                mean_iou,
                dice_coefficient,
                recall,
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
                    filepath=CHECKPOINT_PATH_YNET,
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
