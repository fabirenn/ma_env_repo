import os
import sys

import tensorflow as tf
from deeplab_model import DeepLab
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from keras.callbacks import EarlyStopping

from custom_callbacks import ValidationCallback
from data_loader import create_datasets_for_segnet_training

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

CHECKPOINT_PATH = "artifacts/models/deeplab/deeplab_checkpoint.h5"
LOG_VAL_PRED = "data/predictions/deeplab"


TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"


IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 3

BATCH_SIZE = 4
EPOCHS = 50

train_dataset, val_dataset = create_datasets_for_segnet_training(
    directory_train_images=TRAIN_IMG_PATH,
    directory_train_masks=TRAIN_MASK_PATH,
    directory_val_images=VAL_IMG_PATH,
    directory_val_masks=VAL_MASK_PATH,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    batch_size=BATCH_SIZE,
)


# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="first_deeplab_tests",
    entity="fabio-renn",
    mode="offline",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
)

# [optional] use wandb.config as your config
config = wandb.config


# create model & start training it
model = DeepLab(input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL))

model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
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
            validation_data=val_dataset,
            log_dir=LOG_VAL_PRED,
            apply_crf=True,
        ),
    ],
)