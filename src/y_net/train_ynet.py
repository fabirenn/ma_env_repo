import os
import sys

import keras.metrics
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ynet_model import build_ynet

from custom_callbacks import ValidationCallback, dice_score, specificity_score
from data_loader import create_datasets_for_unet_training
from loss_functions import combined_loss, dice_loss, iou_loss

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

LOG_VAL_PRED = "data/predictions/ynet"
CHECKPOINT_PATH = "artifacts/models/ynet/ynet_checkpoint.h5"

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"'''

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

BATCH_SIZE = 4
EPOCHS = 200
PATIENCE = 70

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

os.environ["WANDB_DIR"] = "wandb/train_ynet"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="ynet",
    entity="fabio-renn",
    mode="offline",
    name="train-ynet",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

# create model & start training it
model = build_ynet(IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, IMG_CHANNEL)

model.compile(
    optimizer="adam",
    loss=combined_loss,
    metrics=[
        "accuracy",
        keras.metrics.BinaryIoU(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        specificity_score,
        dice_score,
    ],
)

model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[
        WandbMetricsLogger(log_freq="epoch"),
        WandbModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss",
            verbose=1,
        ),
        ValidationCallback(
            model=model,
            train_data=train_dataset,
            validation_data=val_dataset,
            log_dir=LOG_VAL_PRED,
            apply_crf=False,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True
        )
    ],
)

wandb.finish()
