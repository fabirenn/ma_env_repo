import os
import sys

import keras
from wandb.integration.keras import WandbMetricsLogger

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deeplab_model import DeepLab

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

CHECKPOINT_PATH = "artifacts/models/deeplab/deeplab_checkpoint.keras"
LOG_VAL_PRED = "data/predictions/deeplab"

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"'''

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

DROPOUT_RATE = 0.3
LEARNING_RATE = 0.0834699
FILTERS = 1024
DILATION_RATES = [3, 6, 9, 18]
USE_BATCHNORM = True
KERNEL_SIZE = 3
INITIALIZER = "he_uniform"
ACTIVATION = "elu"

BATCH_SIZE = 4
EPOCHS = 200
PATIENCE = 70

APPLY_CRF = False

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

os.environ["WANDB_DIR"] = "wandb/train_deeplab"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="deeplab",
    entity="fabio-renn",
    mode="offline",
    name="train-deeplab",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

# create model & start training it
model = DeepLab(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
    dropout_rate=DROPOUT_RATE,
    filters=FILTERS,
    dilation_rates=DILATION_RATES,
    use_batchnorm=USE_BATCHNORM,
    kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
    initializer_function=INITIALIZER,
    activation=ACTIVATION,
)

optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)

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
            apply_crf=APPLY_CRF,
            log_wandb=True,
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
