import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from custom_callbacks import ValidationCallback
from data_loader import (
    convert_to_tensor,
    create_dataset,
    load_images_from_directory,
    load_masks_from_directory,
    make_binary_masks,
    normalize_image_data,
    resize_images,
)
from unet_model_local import unet

TRAIN_IMG_PATH = "data/generated/"
TRAIN_MASK_PATH = "data/train/mask/"
CHECKPOINT_PATH = "artifacts/models/test/"

IMG_WIDTH = 1024
IMG_HEIGHT = 704
IMG_CHANNEL = 3

BATCH_SIZE = 8
EPOCHS = 10

# loading images and masks from their corresponding paths into to separate lists
train_images = load_images_from_directory(TRAIN_IMG_PATH)
train_masks = load_masks_from_directory(TRAIN_MASK_PATH)


# resizing the images to dest size for training
train_images = resize_images(train_images, IMG_WIDTH, IMG_HEIGHT)
train_masks = resize_images(train_masks, IMG_WIDTH, IMG_HEIGHT)

# normalizing the values of the images and binarizing the image masks
train_images = normalize_image_data(train_images)
train_masks = make_binary_masks(train_masks, 30)

# converting the images/masks to tensors + expanding the masks tensor slide to 1 dimension
tensor_train_images = convert_to_tensor(train_images)
tensor_train_masks = convert_to_tensor(train_masks)
tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

# create dataset for training purposes
train_dataset = create_dataset(
    tensor_train_images,
    tensor_train_masks,
    batchsize=BATCH_SIZE,
    buffersize=tf.data.AUTOTUNE,
)


# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="first_unet_tests",
    entity="fabio-renn",
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
    validation_data=train_dataset,
    callbacks=[
        WandbMetricsLogger(log_freq="epoch"),
        WandbModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_best_only=True,
            save_weights_only=True,
        ),
        ValidationCallback(model=model, validation_data=train_dataset),
    ],
)


# Verwendung des Callbacks während des Trainings
# wandb.init(project="mein_projekt", entity="mein_username")
# model.fit(train_images, train_masks, epochs=10, validation_data=(val_images, val_masks),
#           callbacks=[WandbEvalCallback((val_images, val_masks), num_images=10)])
