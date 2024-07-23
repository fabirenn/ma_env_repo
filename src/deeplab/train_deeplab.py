import os
import sys

import keras.metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint
from wandb.integration.keras import WandbMetricsLogger

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from deeplab_model import DeepLab
from custom_callbacks import ValidationCallback
from metrics_calculation import pixel_accuracy, precision, mean_iou, dice_coefficient, recall, f1_score
from data_loader import create_datasets_for_segnet_training
from loss_functions import dice_loss

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

CHECKPOINT_PATH = "artifacts/models/deeplab/deeplab_checkpoint.h5"
LOG_VAL_PRED = "data/predictions/deeplab"

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"'''

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

DROPOUT_RATE = 0.1

BATCH_SIZE = 4
EPOCHS = 200
PATIENCE = 70

APPLY_CRF = False

train_dataset, val_dataset = create_datasets_for_segnet_training(
    directory_train_images=TRAIN_IMG_PATH,
    directory_train_masks=TRAIN_MASK_PATH,
    directory_val_images=VAL_IMG_PATH,
    directory_val_masks=VAL_MASK_PATH,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    batch_size=BATCH_SIZE,
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
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), dropout_rate=DROPOUT_RATE
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

model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[
        WandbMetricsLogger(log_freq="epoch"),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_best_only=True,
            save_weights_only=False,
            monitor="val_loss",
            verbose=1,
        ),
        ValidationCallback(
            validation_data=val_dataset,
            log_dir=LOG_VAL_PRED,
            apply_crf=APPLY_CRF,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True,
        ),
    ],
)

wandb.finish()
