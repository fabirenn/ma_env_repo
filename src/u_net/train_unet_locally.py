import tensorflow as tf
from keras.callbacks import CSVLogger, ModelCheckpoint

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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
from unet_model_local import unet

TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"

CHECKPOINT_PATH = "artifacts/models/test/"

IMG_WIDTH = 1024
IMG_HEIGHT = 1024
IMG_CHANNEL = 8

BATCH_SIZE = 4
EPOCHS = 10

# loading images and masks from their corresponding paths into to separate lists
train_images = load_images_from_directory(TRAIN_IMG_PATH)
train_masks = load_masks_from_directory(TRAIN_MASK_PATH)
val_images = load_images_from_directory(VAL_IMG_PATH)
val_masks = load_masks_from_directory(VAL_MASK_PATH)

# resizing the images to dest size for training
train_images = resize_images(train_images, IMG_WIDTH, IMG_HEIGHT)
train_masks = resize_images(train_masks, IMG_WIDTH, IMG_HEIGHT)
val_images = resize_images(val_images, IMG_WIDTH, IMG_HEIGHT)
val_masks = resize_images(val_masks, IMG_WIDTH, IMG_HEIGHT)

# normalizing the values of the images and binarizing the image masks
train_images = normalize_image_data(train_images)
train_images = preprocess_images(train_images)
train_masks = make_binary_masks(train_masks, 30)

val_images = normalize_image_data(val_images)
val_images = preprocess_images(val_images)
val_masks = make_binary_masks(val_masks, 30)

# converting the images/masks to tensors + expanding the masks tensor slide to
# 1 dimension
tensor_train_images = convert_to_tensor(train_images)
tensor_train_masks = convert_to_tensor(train_masks)
tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

tensor_val_images = convert_to_tensor(val_images)
tensor_val_masks = convert_to_tensor(val_masks)
tensor_val_masks = tf.expand_dims(tensor_val_masks, axis=-1)

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


# create model & start training it
model = unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, BATCH_SIZE)


model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=val_dataset,
    callbacks=[
        CSVLogger(filename="logs/training_metrics.log"),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy",
        ),
        ValidationCallback(model=model, validation_data=val_dataset),
    ],
)
