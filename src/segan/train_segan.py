import os
import sys

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from segan_model import discriminator, generator

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

CHECKPOINT_PATH = "artifacts/models/segan/segan_checkpoint.h5"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

BATCH_SIZE = 4
EPOCHS = 50


generator_model = generator(
    IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, BATCH_SIZE, unet=False
)
discriminator_model = discriminator(
    (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), (IMG_WIDTH, IMG_HEIGHT, 1)
)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator_model,
    discriminator=discriminator_model,
)


def discriminator_loss(real_output, fake_output):
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return loss_fn(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(images, masks):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_masks = generator_model(images, training=True)

        real_output = discriminator_model([images, masks], training=True)
        fake_output = discriminator_model(
            [images, generated_masks], training=True
        )

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator_model.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator_model.trainable_variables
    )

    gen_optimizer.apply_gradients(
        zip(gradients_of_generator, generator_model.trainable_variables)
    )
    disc_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator_model.trainable_variables)
    )

    return gen_loss, disc_loss


def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch, mask_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch, mask_batch)

        print(
            f"Epoch {epoch+1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}"
        )


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
train_masks = make_binary_masks(train_masks, 30)
print("Train masks binarized..")

val_images = normalize_image_data(val_images)
print("Val images normalized..")
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

train(train_dataset, epochs=50)
