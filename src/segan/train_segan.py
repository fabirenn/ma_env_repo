import os
import sys
import keras
import keras.backend
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from segan_model import discriminator, vgg_model

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from metrics_calculation import pixel_accuracy, precision, mean_iou, dice_coefficient, recall, f1_score
from custom_callbacks import ValidationCallback
from data_loader import (
    create_datasets_for_unet_training,
)
from processing import safe_predictions_locally
from loss_functions import discriminator_loss, generator_loss

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"


"""
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"
"""


LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

DROPOUT_RATE = 0.1
BATCH_SIZE = 4
EPOCHS = 200

GENERATOR_TRAINING_STEPS = 5

PATIENCE = 70
BEST_IOU = 0
WAIT = 0


os.environ["WANDB_DIR"] = "wandb/train_segan"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="segan",
    entity="fabio-renn",
    mode="offline",
    name="train-segan",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

keras.backend.set_image_data_format("channels_last")

generator_model = sm.Unet(
    backbone_name="resnet34",
    input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
    classes=5,
    activation="softmax",
    encoder_weights=None,
    encoder_features="default",
    decoder_block_type="upsampling",
    decoder_filters=(256, 128, 64, 32, 16),
    decoder_use_batchnorm=True,
)

generator_model.summary()


discriminator_model = discriminator(
    (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), (IMG_WIDTH, IMG_HEIGHT, 5)
)

# loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# loss_fn = combined_loss
gen_optimizer = keras.optimizers.Adam(1e-4)
disc_optimizer = keras.optimizers.Adam(1e-4)
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator_model,
    discriminator=discriminator_model,
)


def evaluate_generator(generator, dataset):
    # Implement the evaluation logic
    accuracy_value = 0.0
    pixel_accuracy_value = 0.0
    precision_value_value = 0.0
    mean_iou_value = 0.0
    dice_value = 0.0
    f1_value = 0.0
    recall_value = 0.0

    # Calculate metrics over the validation dataset
    for image_batch, mask_batch in dataset:
        predictions = generator(image_batch, training=False)
        accuracy_value += keras.metrics.Accuracy()(mask_batch, predictions)
        pixel_accuracy_value += pixel_accuracy(mask_batch, predictions)
        precision_value_value += precision(mask_batch, predictions)
        mean_iou_value += mean_iou(mask_batch, predictions)
        dice_value += dice_coefficient(mask_batch, predictions)
        f1_value += f1_score(mask_batch, predictions)
        recall_value += recall(mask_batch, predictions)

    # Average the metrics over the dataset
    accuracy_value /= len(dataset)
    pixel_accuracy_value /= len(dataset)
    precision_value_value /= len(dataset)
    mean_iou_value /= len(dataset)
    dice_value /= len(dataset)
    f1_value /= len(dataset)
    recall_value /= len(dataset)
    return accuracy_value, pixel_accuracy_value, precision_value_value, mean_iou_value, dice_value, f1_value, recall_value


@tf.function
def train_step_generator(images, masks):

    with tf.GradientTape() as gen_tape:
        generated_masks = generator_model(images, training=True)
        fake_output = discriminator_model(
            [images, generated_masks], training=True
        )
        gen_loss = generator_loss(fake_output)
    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator_model.trainable_variables
    )
    gen_optimizer.apply_gradients(
        zip(gradients_of_generator, generator_model.trainable_variables)
    )
    return gen_loss


@tf.function
def train_step_discriminator(images, masks):

    with tf.GradientTape() as disc_tape:
        generated_masks = generator_model(images, training=True)
        real_output = discriminator_model([images, masks], training=True)
        fake_output = discriminator_model(
            [images, generated_masks], training=True
        )
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator_model.trainable_variables
    )
    disc_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator_model.trainable_variables)
    )
    return disc_loss


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


def train(train_dataset, val_dataset, epochs, trainingsteps):
    global BEST_IOU, WAIT
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        for image_batch, mask_batch in train_dataset:
            for _ in range(trainingsteps):
                gen_loss = train_step_generator(image_batch, mask_batch)
                # clip_discriminator_weights(discriminator_model, clip_value)
            disc_loss = train_step_discriminator(image_batch, mask_batch)

        (
            train_accuracy,
            train_pixel_accuracy,
            train_precision,
            train_mean_iou,
            train_dice_coefficient,
            train_f1,
            train_recall,
        ) = evaluate_generator(generator_model, train_dataset)
        (
            val_accuracy,
            val_pixel_accuracy,
            val_precision,
            val_mean_iou,
            val_dice_coefficient,
            val_f1,
            val_recall,
        ) = evaluate_generator(generator_model, val_dataset)

        wandb.log(
            {
                "epoch": epoch + 1,
                "gen_loss": gen_loss,
                "disc_loss": disc_loss,
                "train_accuracy": train_accuracy,
                "train_pixel_accuracy": train_pixel_accuracy,
                "train_precision": train_precision,
                "train_mean_iou": train_mean_iou,
                "train_dice_coefficient": train_dice_coefficient,
                "train_f1": train_f1,
                "train_recall": train_recall,
                "val_accuracy": val_accuracy,
                "val_pixel_accuracy": val_pixel_accuracy,
                "val_precision": val_precision,
                "val_mean_iou": val_mean_iou,
                "val_dice_coefficient": val_dice_coefficient,
                "val_f1": val_f1,
                "val_recall": val_recall
            }
        )

        # Print the losses and metrics
        print(
            f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f}"
        )
        print(
            f"Train Metrics - Accuracy: {train_accuracy:.4f}, PA: {train_pixel_accuracy:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, Mean-IOU: {train_mean_iou:.4f}, Dice: {train_dice_coefficient:.4f}, F1: {train_f1:.4f}"
        )
        print(
            f"Validation Metrics - Accuracy: {val_accuracy:.4f}, PA: {val_pixel_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, IOU: {val_mean_iou:.4f}, Dice: {val_dice_coefficient:.4f}, F1: {val_f1:.4f}"
        )

        generate_images(model=generator_model, dataset=val_dataset, epoch=epoch)

        if val_mean_iou > BEST_IOU:
            BEST_IOU = val_mean_iou
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


train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=EPOCHS,
    trainingsteps=5,
)

wandb.finish()
