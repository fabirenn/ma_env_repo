import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping
from segan_model import discriminator, generator, vgg_model
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_callbacks import ValidationCallback
from data_loader import (
    create_datasets_for_segnet_training,
    create_datasets_for_unet_training,
    create_testdataset_for_segnet_training,
    create_testdataset_for_unet_training,
)
from loss_functions import combined_loss, dice_loss, iou_loss
from processing import safe_predictions_locally

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"
TEST_IMG_PATH = "data/training_test/images_mixed"
TEST_MASK_PATH = "data/training_test/labels_mixed"

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"
TEST_IMG_PATH = "data/local/test/images"
TEST_MASK_PATH = "data/local/test/labels"'''


LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.h5"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

BATCH_SIZE = 4
EPOCHS = 100
UNET = True

PATIENCE = 80
MIN_DELTA_LOSS = 0.01
BEST_GEN_LOSS = np.inf
WAIT = 0


generator_model = generator(IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE, used_unet=UNET)

discriminator_model = discriminator(
    (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), (IMG_WIDTH, IMG_HEIGHT, 1)
)

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# loss_fn = combined_loss
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator_model,
    discriminator=discriminator_model,
)

vgg_model = vgg_model()


def discriminator_loss(real_output, fake_output):
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return loss_fn(tf.ones_like(fake_output), fake_output)


def convert_grayscale_to_rgb(images):
    return tf.image.grayscale_to_rgb(images)


def extract_features(model, images):
    rgb_images = convert_grayscale_to_rgb(images)
    return model(rgb_images)


def multi_scale_feature_loss(real_images, generated_images, feature_extractor):
    real_features = extract_features(feature_extractor, real_images)
    generated_features = extract_features(feature_extractor, generated_images)
    loss = 0
    for real, generated in zip(real_features, generated_features):
        loss += tf.reduce_mean(tf.abs(real - generated))
    return loss


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

        ms_feature_loss = multi_scale_feature_loss(
            masks, generated_masks, vgg_model
        )
        total_gen_loss = gen_loss + ms_feature_loss

    gradients_of_generator = gen_tape.gradient(
        total_gen_loss, generator_model.trainable_variables
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

    return total_gen_loss, disc_loss


def generate_images(model, dataset, epoch):
    sample = dataset.take(1)
    image_batch, mask_batch = next(iter(sample))
    pred_batch = model.predict(image_batch)

    safe_predictions_locally(
        range=None,
        iterator=epoch,
        test_images=image_batch[0],
        predictions=(pred_batch[0] > 0.5).astype(np.uint8),
        test_masks=mask_batch[0],
        pred_img_path=LOG_VAL_PRED,
        val=True,
    )


def train(train_dataset, test_dataset, epochs):
    global BEST_GEN_LOSS, WAIT
    for epoch in range(epochs):
        for image_batch, mask_batch in train_dataset:
            gen_loss, disc_loss = train_step(image_batch, mask_batch)

        print(
            f"Epoch {epoch+1}, Gen Loss: {gen_loss.numpy()}, Disc Loss: {disc_loss.numpy()}"
        )
        generate_images(
            model=generator_model, dataset=test_dataset, epoch=epoch
        )

        if gen_loss < BEST_GEN_LOSS - MIN_DELTA_LOSS:
            BEST_GEN_LOSS = gen_loss
            checkpoint.save(file_prefix=CHECKPOINT_PATH)
            generator_model.save(CHECKPOINT_PATH)
            print("Improved & Saved model")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping triggered")
                return
            else:
                print("Waiting for improvement..")


if UNET is True:
    train_dataset, val_dataset = create_datasets_for_unet_training(
        directory_train_images=TRAIN_IMG_PATH,
        directory_train_masks=TRAIN_MASK_PATH,
        directory_val_images=VAL_IMG_PATH,
        directory_val_masks=VAL_MASK_PATH,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        batch_size=BATCH_SIZE,
    )

    test_dataset, test_images, test_masks = (
        create_testdataset_for_unet_training(
            directory_test_images=TEST_IMG_PATH,
            directory_test_masks=TEST_MASK_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            batch_size=BATCH_SIZE,
        )
    )

else:
    train_dataset, val_dataset = create_datasets_for_segnet_training(
        directory_train_images=TRAIN_IMG_PATH,
        directory_train_masks=TRAIN_MASK_PATH,
        directory_val_images=VAL_IMG_PATH,
        directory_val_masks=VAL_MASK_PATH,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
        batch_size=BATCH_SIZE,
    )

    test_dataset, test_images, test_masks = (
        create_testdataset_for_segnet_training(
            directory_test_images=TEST_IMG_PATH,
            directory_test_masks=TEST_MASK_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            batch_size=BATCH_SIZE,
        )
    )

combined_dataset = train_dataset.concatenate(val_dataset)

train(train_dataset=combined_dataset, test_dataset=test_dataset, epochs=EPOCHS)
