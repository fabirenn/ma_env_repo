import os
import sys

import keras
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ynet_model import build_ynet, build_feature_extractor_for_pretraining, build_ynet_with_pretrained_semantic_extractor

from custom_callbacks import ValidationCallback
from data_loader import create_datasets_for_unet_training
from loss_functions import dice_loss
from metrics_calculation import (
    dice_coefficient,
    f1_score,
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
CHECKPOINT_PATH = "artifacts/models/ynet/ynet_checkpoint.keras"

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"'''

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

DROPOUT_RATE = 0.0
BATCH_SIZE = 8
EPOCHS = 30
PATIENCE = 10

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
semantic_extractor_model = build_feature_extractor_for_pretraining(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE)
semantic_extractor_model.load_weights(CHECKPOINT_PATH)

model = build_ynet_with_pretrained_semantic_extractor(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE, semantic_extractor_model)

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
