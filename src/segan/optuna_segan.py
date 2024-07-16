import os
import sys

import keras.backend
import numpy as np
import optuna
import segmentation_models as sm
import tensorflow as tf
from segan_model import discriminator, vgg_model
from train_segan import (
    convert_grayscale_to_rgb,
    discriminator_loss,
    evaluate_generator,
    extract_features,
    generator_loss,
    multi_scale_feature_loss,
)

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_callbacks import clear_directory
from data_loader import (
    create_datasets_for_segnet_training,
    create_datasets_for_unet_training,
)
from loss_functions import dice_loss, iou_loss
from processing import safe_predictions_locally

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.h5"

IMG_WIDTH = 128
IMG_HEIGHT = 128

EPOCHS = 50
PATIENCE = 30
BEST_IOU = 0
WAIT = 0

keras.backend.set_image_data_format("channels_last")

os.environ["WANDB_DIR"] = "wandb/train_segan"
os.environ["WANDB_DATA_DIR"] = "/work/fi263pnye-ma_data/tmp"


def objective(trial):
    # Define hyperparameters to tune
    BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8, 12, 16])
    IMG_CHANNEL = trial.suggest_categorical("img_channel", [3, 8])
    BACKBONE = trial.suggest_categorical(
        "backbone", ["resnet34", "resnet50", "efficientnetb0"]
    )
    GENERATOR_TRAINING_STEPS = trial.suggest_int("g_training_steps", 0, 9)
    LEARNING_RATE = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)

    tf.keras.backend.clear_session()

    # Initialize Wandb
    wandb.init(
        project="segan",
        entity="fabio-renn",
        mode="offline",
        name=f"train-segan-{trial.number}",
        config={
            "metric": "accuracy",
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
        },
        dir=os.environ["WANDB_DIR"],
    )

    generator_model = sm.Unet(
        backbone_name=BACKBONE,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
        classes=1,
        activation="sigmoid",
        encoder_weights=None,
        encoder_features="default",
        decoder_block_type="upsampling",
        decoder_filters=(256, 128, 64, 32, 16),
        decoder_use_batchnorm=True,
    )

    discriminator_model = discriminator(
        (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), (IMG_WIDTH, IMG_HEIGHT, 1)
    )

    gen_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    disc_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=gen_optimizer,
        discriminator_optimizer=disc_optimizer,
        generator=generator_model,
        discriminator=discriminator_model,
    )

    vgg_model = vgg_model()

    def train_step_generator(images, masks):
        with tf.GradientTape() as gen_tape:
            generated_masks = generator_model(images, training=True)
            fake_output = discriminator_model(
                [images, generated_masks], training=True
            )
            gen_loss = generator_loss(fake_output)
            ms_feature_loss = multi_scale_feature_loss(
                masks, generated_masks, vgg_model
            )
            total_gen_loss = gen_loss + ms_feature_loss
        gradients_of_generator = gen_tape.gradient(
            total_gen_loss, generator_model.trainable_variables
        )
        gen_optimizer.apply_gradients(
            zip(gradients_of_generator, generator_model.trainable_variables)
        )
        return gen_loss

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
            zip(
                gradients_of_discriminator,
                discriminator_model.trainable_variables,
            )
        )
        return disc_loss

    def train(train_dataset, val_dataset, epochs, trainingsteps):
        global BEST_IOU, WAIT
        best_gen_loss = float("inf")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{EPOCHS}")
            for image_batch, mask_batch in train_dataset:
                for _ in range(trainingsteps):
                    gen_loss = train_step_generator(image_batch, mask_batch)
                disc_loss = train_step_discriminator(image_batch, mask_batch)
            train_metrics = evaluate_generator(generator_model, train_dataset)
            val_metrics = evaluate_generator(generator_model, val_dataset)
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "gen_loss": gen_loss,
                    "disc_loss": disc_loss,
                    "train_accuracy": train_metrics[0],
                    "train_iou": train_metrics[1],
                    "train_precision": train_metrics[2],
                    "train_recall": train_metrics[3],
                    "train_specificity": train_metrics[4],
                    "train_dice": train_metrics[5],
                    "val_accuracy": val_metrics[0],
                    "val_iou": val_metrics[1],
                    "val_precision": val_metrics[2],
                    "val_recall": val_metrics[3],
                    "val_specificity": val_metrics[4],
                    "val_dice": val_metrics[5],
                }
            )
            print(
                f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f}"
            )
            print(
                f"Train Metrics - Accuracy: {train_metrics[0]:.4f}, IoU: {train_metrics[1]:.4f}, Precision: {train_metrics[2]:.4f}, Recall: {train_metrics[3]:.4f}, Specificity: {train_metrics[4]:.4f}, Dice: {train_metrics[5]:.4f}"
            )
            print(
                f"Validation Metrics - Accuracy: {val_metrics[0]:.4f}, IoU: {val_metrics[1]:.4f}, Precision: {val_metrics[2]:.4f}, Recall: {val_metrics[3]:.4f}, Specificity: {val_metrics[4]:.4f}, Dice: {val_metrics[5]:.4f}"
            )

            if gen_loss < best_gen_loss:
                best_gen_loss = gen_loss
                checkpoint.save(file_prefix=CHECKPOINT_PATH)
                generator_model.save(CHECKPOINT_PATH)
                print("Improved & Saved model\n")
                WAIT = 0
            else:
                WAIT += 1
                if WAIT >= PATIENCE:
                    print("Early stopping triggered\n")
                    return best_gen_loss

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

    best_gen_loss = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        trainingsteps=GENERATOR_TRAINING_STEPS,
    )

    wandb.finish()

    clear_directory("/work/fi263pnye-ma_data/tmp/artifacts")

    return best_gen_loss


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
