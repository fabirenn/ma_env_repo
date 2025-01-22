import os
import sys

import keras
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet_model import unet

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

LOG_VAL_PRED = "data/predictions/unet"
CHECKPOINT_PATH = "artifacts/models/unet/unet_checkpoint.keras"

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"'''

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

DROPOUT_RATE = 0.0
LEARNING_RATE = 0.0010591722685292172

BATCH_SIZE = 12
EPOCHS = 200
PATIENCE = 50

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

os.environ["WANDB_DIR"] = "wandb/train_unet"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="unet",
    entity="fabio-renn",
    mode="offline",
    name="train-unet",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

filters_list = [16, 32, 64, 128, 256, 512]

# create model & start training it
model = unet(
    IMG_WIDTH,
    IMG_HEIGHT,
    IMG_CHANNEL,
    DROPOUT_RATE,
    filters_list,
    kernel_size=(3, 3),
    activation="elu",
    use_batchnorm=True,
    initializer_function="he_normal",
    training=True,
)

optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)

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
            log_wandb=True
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
