import os
import sys

import tensorflow as tf
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet_architecture_hcp import unet

from custom_callbacks import ValidationCallback
from data_loader import (
    convert_to_tensor,
    create_dataset,
    load_images_from_directory,
    load_masks_from_directory,
    make_binary_masks,
    normalize_image_data,
    preprocess_images,
    resize_images,
)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

CHECKPOINT_PATH = "artifacts/models/unet/unet_checkpoint.h5"

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

IMG_CHANNEL = 8

BATCH_SIZE = 4
EPOCHS = 50


# loading images and masks from their corresponding paths into to separate lists
train_images = load_images_from_directory(TRAIN_IMG_PATH)
train_masks = load_masks_from_directory(TRAIN_MASK_PATH)
print("Train-Images successfully loaded..")
val_images = load_images_from_directory(VAL_IMG_PATH)
val_masks = load_masks_from_directory(VAL_MASK_PATH)
print("Validation-Data successfully loaded..")

# resizing the images to dest size for training
train_images = resize_images(train_images, IMG_WIDTH, IMG_HEIGHT)
train_masks = resize_images(train_masks, IMG_WIDTH, IMG_HEIGHT)
val_images = resize_images(val_images, IMG_WIDTH, IMG_HEIGHT)
val_masks = resize_images(val_masks, IMG_WIDTH, IMG_HEIGHT)
print("All images resized..")

# normalizing the values of the images and binarizing the image masks
train_images = normalize_image_data(train_images)
print("Train images normalized..")
train_images = preprocess_images(train_images)
print("Train images preprocessed..")
train_masks = make_binary_masks(train_masks, 30)
print("Train masks binarized..")

val_images = normalize_image_data(val_images)
print("Val images normalized..")
val_images = preprocess_images(val_images)
print("Val images preprocessed..")
val_masks = make_binary_masks(val_masks, 30)
print("Val masks binarized..")

# converting the images/masks to tensors + expanding the masks tensor slide to
# 1 dimension
tensor_train_images = convert_to_tensor(train_images)
tensor_train_masks = convert_to_tensor(train_masks)
tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)


tensor_val_images = convert_to_tensor(val_images)
tensor_val_masks = convert_to_tensor(val_masks)
tensor_val_masks = tf.expand_dims(tensor_val_masks, axis=-1)

print("Everything converted to tensors..")

# create dataset for training purposes
train_dataset = create_dataset(
    tensor_train_images,
    tensor_train_masks,
    batchsize=BATCH_SIZE,
    buffersize=tf.data.AUTOTUNE,
)

val_dataset = create_dataset(
    tensor_val_images,
    tensor_val_masks,
    batchsize=BATCH_SIZE,
    buffersize=tf.data.AUTOTUNE,
)

print("Train and Val DataSet created..")


# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="first_unet_tests",
    entity="fabio-renn",
    mode="offline",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
)

# [optional] use wandb.config as your config
config = wandb.config


# create model & start training it
model = unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, BATCH_SIZE)

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
            log_dir="data/predictions/unet",
        ),
        EarlyStopping(monitor="val_loss", mode="auto", patience=6),
    ],
)
