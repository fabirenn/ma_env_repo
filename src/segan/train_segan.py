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

from custom_callbacks import ValidationCallback, dice_score, specificity_score
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


TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"
TEST_IMG_PATH = "data/local/test/images"
TEST_MASK_PATH = "data/local/test/labels"


LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.h5"

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNEL = 8

BATCH_SIZE = 4
EPOCHS = 100
UNET = True

PATIENCE = 80
MIN_DELTA_LOSS = 0.01
BEST_GEN_LOSS = np.inf
WAIT = 0


os.environ["WANDB_DIR"] = "wandb/train_segan"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="segan",
    entity="fabio-renn",
    mode="offline",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

generator_model = generator(
    IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, BATCH_SIZE, used_unet=UNET
)



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

generator_model.compile(
    optimizer=gen_optimizer,
    loss=loss_fn,
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),
    ]
)


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


def evaluate_generator(generator, dataset):
    # Implement the evaluation logic
    accuracy = 0.0
    iou = 0.0
    precision = 0.0
    recall = 0.0
    specificity = 0.0
    dice = 0.0
    # Calculate metrics over the validation dataset
    for image_batch, mask_batch in dataset:
        predictions = generator(image_batch, training=False)
        accuracy += tf.keras.metrics.BinaryAccuracy()(mask_batch, predictions)
        iou += tf.keras.metrics.BinaryIoU()(mask_batch, predictions)
        precision += tf.keras.metrics.Precision()(mask_batch, predictions)
        recall += tf.keras.metrics.Recall()(mask_batch, predictions)
        specificity += specificity_score(mask_batch, predictions)
        dice += dice_score(mask_batch, predictions)

    # Average the metrics over the dataset
    accuracy /= len(dataset)
    iou /= len(dataset)
    precision /= len(dataset)
    recall /= len(dataset)
    specificity /= len(dataset)
    dice /= len(dataset)
    return accuracy, iou, precision, recall, specificity, dice


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


def train(train_dataset, val_dataset, epochs):
    global BEST_GEN_LOSS, WAIT
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        for image_batch, mask_batch in train_dataset:
            gen_loss, disc_loss = train_step(image_batch, mask_batch)

        (
            train_accuracy,
            train_iou,
            train_precision,
            train_recall,
            train_specificity,
            train_dice,
        ) = evaluate_generator(generator_model, train_dataset)
        (
            val_accuracy,
            val_iou,
            val_precision,
            val_recall,
            val_specificity,
            val_dice,
        ) = evaluate_generator(generator_model, val_dataset)

        wandb.log(
            {
                "epoch": epoch + 1,
                "gen_loss": gen_loss,
                "disc_loss": disc_loss,
                "train_accuracy": train_accuracy,
                "train_iou": train_iou,
                "train_precision": train_precision,
                "train_recall": train_recall,
                "train_specificity": train_specificity,
                "train_dice": train_dice,
                "val_accuracy": val_accuracy,
                "val_iou": val_iou,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_specificity": val_specificity,
                "val_dice": val_dice,
            }
        )

        # Print the losses and metrics
        print(
            f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f}"
        )
        print(
            f"Train Metrics - Accuracy: {train_accuracy:.4f}, IoU: {train_iou:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Specificity: {train_specificity:.4f}, Dice: {train_dice:.4f}"
        )
        print(
            f"Validation Metrics - Accuracy: {val_accuracy:.4f}, IoU: {val_iou:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, Specificity: {val_specificity:.4f}, Dice: {val_dice:.4f}"
        )

        generate_images(model=generator_model, dataset=val_dataset, epoch=epoch)

        if gen_loss < BEST_GEN_LOSS - MIN_DELTA_LOSS:
            BEST_GEN_LOSS = gen_loss
            checkpoint.save(file_prefix=CHECKPOINT_PATH)
            generator_model.save(CHECKPOINT_PATH)
            print("Improved & Saved model\n")
            wait = 0
        else:
            wait += 1
            if wait >= PATIENCE:
                print("Early stopping triggered\n")
                return
            else:
                print("Waiting for improvement..\n")


if UNET is True:
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

train(train_dataset=train_dataset, val_dataset=val_dataset, epochs=EPOCHS)

wandb.finish()