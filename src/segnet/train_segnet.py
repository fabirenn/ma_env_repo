import os
import sys

import keras
from segnet_model import segnet
from wandb.integration.keras import WandbCallback, WandbMetricsLogger

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_callbacks import ValidationCallback
from data_loader import create_datasets_for_segnet_training
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

CHECKPOINT_PATH = "artifacts/models/segnet/segnet_checkpoint.keras"
LOG_VAL_PRED = "data/predictions/segnet"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

DROPOUT_RATE = 0.1
BATCH_SIZE = 4
EPOCHS = 50
PATIENCE = 30

train_dataset, val_dataset = create_datasets_for_segnet_training(
    directory_train_images=TRAIN_IMG_PATH,
    directory_train_masks=TRAIN_MASK_PATH,
    directory_val_images=VAL_IMG_PATH,
    directory_val_masks=VAL_MASK_PATH,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    batch_size=BATCH_SIZE,
)

os.environ["WANDB_DIR"] = "wandb/train_segnet"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="segnet",
    entity="fabio-renn",
    mode="offline",
    name="train-segnet",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

# create model & start training it
model = segnet(
    input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
    dropout_rate=DROPOUT_RATE,
    num_filters=[32, 64, 128, 256],
    kernel_size=(3, 3),
    activation="relu",
    use_batchnorm=True,
    initializer_function="he_normal"
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
        recall,
    ],
)

model.fit(
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
            log_wandb=False
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
    ],
)

wandb.finish()
