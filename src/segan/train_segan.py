import os
import sys

import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from segan_model import discriminator, generator
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_callbacks import ValidationCallback
from data_loader import (
    create_datasets_for_segnet_training,
    create_datasets_for_unet_training,
)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

'''TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"'''

TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"

CHECKPOINT_PATH = "artifacts/models/segan/segan_checkpoint.h5"

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNEL = 3

BATCH_SIZE = 4
EPOCHS = 10


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


train_dataset, val_dataset = create_datasets_for_segnet_training(
    directory_train_images=TRAIN_IMG_PATH,
    directory_train_masks=TRAIN_MASK_PATH,
    directory_val_images=VAL_IMG_PATH,
    directory_val_masks=VAL_MASK_PATH,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    batch_size=BATCH_SIZE,
)

train(train_dataset, epochs=50)
