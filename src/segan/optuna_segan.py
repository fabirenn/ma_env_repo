import os
import sys

import keras.backend
import numpy as np
import optuna
import segmentation_models as sm
import tensorflow as tf
from segan_model import discriminator

import wandb

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "u_net"))
)
from unet_model import unet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_callbacks import ValidationCallback
from data_loader import create_datasets_for_unet_training
from loss_functions import (
    combined_discriminator_loss,
    combined_generator_loss,
    discriminator_loss,
    generator_loss,
)
from metrics_calculation import (
    accuracy,
    dice_coefficient,
    f1_score,
    mean_iou,
    pixel_accuracy,
    precision,
    recall,
)
from processing import safe_predictions_locally

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.keras"

IMG_WIDTH = 256
IMG_HEIGHT = 256

EPOCHS = 50
PATIENCE = 30
BEST_IOU = 0
WAIT = 0

os.environ["WANDB_DIR"] = "wandb/train_segan"
os.environ["WANDB_DATA_DIR"] = "/work/fi263pnye-ma_data/tmp"


def objective(trial):
    BATCH_SIZE = trial.suggest_categorical(
        "batch_size", [8, 12, 16, 20, 24, 28, 32]
    )
    IMG_CHANNEL = trial.suggest_categorical("img_channel", [3, 8])
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.1)
    GENERATOR_TRAINING_STEPS = trial.suggest_int("g_training_steps", 3, 10)
    FILTERS_DEPTH = trial.suggest_int("filters_depth", 3, 6)

    filters_list = [16, 32, 64, 128, 256, 512, 1024]  # Base list of filters
    discriminator_filters = filters_list[:FILTERS_DEPTH]

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
    generator_model = unet(
        IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE, discriminator_filters
    )

    discriminator_model = discriminator(
        (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
        (IMG_WIDTH, IMG_HEIGHT, 5),
        discriminator_filters,
    )
    # Create the intermediate model
    intermediate_layer_model = keras.Model(
    inputs=discriminator_model.input,
    outputs=[layer.output for layer in discriminator_model.layers if 'conv' in layer.name or 'bn' in layer.name]
)

    gen_optimizer = keras.optimizers.Adam(1e-4)
    disc_optimizer = keras.optimizers.Adam(1e-4)
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=gen_optimizer,
        discriminator_optimizer=disc_optimizer,
        generator=generator_model,
        discriminator=discriminator_model,
    )

    generator_model.summary()
    discriminator_model.summary()

    generator_model.compile(
        optimizer=gen_optimizer, loss=generator_loss, metrics=["accuracy"]
    )

    discriminator_model.compile(
        optimizer=disc_optimizer, loss=discriminator_loss, metrics=["accuracy"]
    )

    def evaluate_generator(generator, dataset):
        metrics = {
            "accuracy": keras.metrics.Mean(name="accuracy"),
            "dice": keras.metrics.Mean(name="dice"),
            "mean_iou": keras.metrics.Mean(name="mean_iou"),
            "pixel_accuracy": keras.metrics.Mean(name="pixel_accuracy"),
            "precision": keras.metrics.Mean(name="precision"),
            "recall": keras.metrics.Mean(name="recall"),
            "f1": keras.metrics.Mean(name="f1"),
        }

        # Calculate metrics over the validation dataset
        for image_batch, mask_batch in dataset:
            predictions = generator(image_batch, training=False)
            metrics["accuracy"].update_state(accuracy(mask_batch, predictions))
            metrics["dice"].update_state(
                dice_coefficient(mask_batch, predictions)
            )
            metrics["mean_iou"].update_state(mean_iou(mask_batch, predictions))
            metrics["pixel_accuracy"].update_state(
                pixel_accuracy(mask_batch, predictions)
            )
            metrics["precision"].update_state(
                precision(mask_batch, predictions)
            )
            metrics["recall"].update_state(recall(mask_batch, predictions))
            metrics["f1"].update_state(f1_score(mask_batch, predictions))

        results = {
            name: metric.result().numpy() for name, metric in metrics.items()
        }
        return results

    @tf.function
    def train_step_generator(images, masks):
        with tf.GradientTape() as gen_tape:
            generated_masks = generator_model(images, training=True)
            gen_loss = combined_generator_loss(discriminator_model, intermediate_layer_model, images, masks, generated_masks)
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
            disc_loss = combined_discriminator_loss(discriminator_model, intermediate_layer_model, images, masks, generated_masks)
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
        global WAIT
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
                    "train_accuracy": train_metrics["accuracy"],
                    "train_pixel_accuracy": train_metrics["pixel_accuracy"],
                    "train_precision": train_metrics["precision"],
                    "train_mean_iou": train_metrics["mean_iou"],
                    "train_dice_coefficient": train_metrics["dice"],
                    "train_f1": train_metrics["f1"],
                    "train_recall": train_metrics["recall"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                    "val_precision": val_metrics["precision"],
                    "val_mean_iou": val_metrics["mean_iou"],
                    "val_dice_coefficient": val_metrics["dice"],
                    "val_f1": val_metrics["f1"],
                    "val_recall": val_metrics["recall"],
                }
            )
            print(
                f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f}"
            )
            print(
                f"Train Metrics - Accuracy: {train_metrics['accuracy']:.4f}, PA: {train_metrics['pixel_accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, IOU: {train_metrics['mean_iou']:.4f}, Dice: {train_metrics['dice']:.4f}, F1: {train_metrics['f1']:.4f}"
            )
            print(
                f"Validation Metrics - Accuracy: {val_metrics['accuracy']:.4f}, PA: {val_metrics['pixel_accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, IOU: {val_metrics['mean_iou']:.4f}, Dice: {val_metrics['dice']:.4f}, F1: {val_metrics['f1']:.4f}"
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

    best_gen_loss = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        trainingsteps=GENERATOR_TRAINING_STEPS,
    )

    wandb.finish()

    return best_gen_loss


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    # clear_directory("/work/fi263pnye-ma_data/tmp/artifacts")

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
