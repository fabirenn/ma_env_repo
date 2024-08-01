import os
import sys

import keras.backend
import numpy as np
import optuna
import segmentation_models as sm
import tensorflow as tf
from segan_model import discriminator

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_callbacks import (
    ValidationCallback,
)
from metrics_calculation import pixel_accuracy, precision, mean_iou, dice_coefficient, recall, f1_score
from data_loader import (
    create_datasets_for_unet_training,
)
from loss_functions import discriminator_loss, generator_loss
from processing import safe_predictions_locally


os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512

EPOCHS = 50
PATIENCE = 30
BEST_IOU = 0
WAIT = 0

os.environ["WANDB_DIR"] = "wandb/train_segan"
os.environ["WANDB_DATA_DIR"] = "/work/fi263pnye-ma_data/tmp"


def objective(trial):
    BATCH_SIZE = trial.suggest_categorical("batch_size", [8, 12, 16, 20, 24, 28, 32])
    IMG_CHANNEL = trial.suggest_categorical("img_channel", [3, 8])
    BACKBONE = trial.suggest_categorical(
        "backbone", ["resnet34", "resnet50", "efficientnetb0"]
    )
    GENERATOR_TRAINING_STEPS = trial.suggest_int("g_training_steps", 2, 7)
    FILTERS_DEPTH = trial.suggest_int("filters_depth", 3, 6)

    filters_list = [16, 32, 64, 128, 256, 512, 1024]  # Base list of filters
    decoder_filters = filters_list[-FILTERS_DEPTH:][::-1]  # Slice and reverse the list
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

    generator_model = sm.Unet(
        backbone_name=BACKBONE,
        input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
        classes=5,
        activation="softmax",
        encoder_weights=None,
        encoder_features="default",
        decoder_block_type="transpose",
        decoder_filters=decoder_filters,
        decoder_use_batchnorm=True,
    )

    discriminator_model = discriminator(
        (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL), (IMG_WIDTH, IMG_HEIGHT, 5), discriminator_filters
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
        optimizer=gen_optimizer,
        loss=generator_loss,
        metrics=['accuracy']
    )

    discriminator_model.compile(
        optimizer=disc_optimizer,
        loss=discriminator_loss,
        metrics=['accuracy']
    )

    def evaluate_generator(generator, dataset):
        # Implement the evaluation logic
        accuracy_metric = keras.metrics.Accuracy()
        accuracy_metric.reset_state()
        pixel_accuracy_value = 0.0
        precision_value_value = 0.0
        mean_iou_value = 0.0
        dice_value = 0.0
        f1_value = 0.0
        recall_value = 0.0

    # Calculate metrics over the validation dataset
        for image_batch, mask_batch in dataset:
            predictions = generator(image_batch, training=False)
            accuracy_metric.update_state(mask_batch, predictions)

            pixel_accuracy_value += pixel_accuracy(mask_batch, predictions)
            precision_value_value += precision(mask_batch, predictions)
            mean_iou_value += mean_iou(mask_batch, predictions)
            dice_value += dice_coefficient(mask_batch, predictions)
            f1_value += f1_score(mask_batch, predictions)
            recall_value += recall(mask_batch, predictions)

        # Average the metrics over the dataset
        accuracy_value = accuracy_metric.result().numpy()
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
            gen_loss = generator_loss(fake_output, generated_masks, masks)
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
                    "train_pixel_accuracy": train_metrics[1],
                    "train_precision": train_metrics[2],
                    "train_mean_iou": train_metrics[3],
                    "train_dice_coefficient": train_metrics[4],
                    "train_f1": train_metrics[5],
                    "train_recall": train_metrics[6],
                    "val_accuracy": val_metrics[0],
                    "val_pixel_accuracy": val_metrics[1],
                    "val_precision": val_metrics[2],
                    "val_mean_iou": val_metrics[3],
                    "val_dice_coefficient": val_metrics[4],
                    "val_f1": val_metrics[5],
                    "val_recall": val_metrics[6]
                }
            )
            print(
                f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f}"
            )
            print(
                f"Train Metrics - Accuracy: {train_metrics[0]:.4f}, PA: {train_metrics[1]:.4f}, Precision: {train_metrics[2]:.4f}, MeanIOU: {train_metrics[3]:.4f}, Dice: {train_metrics[4]:.4f}, F1: {train_metrics[5]:.4f}, Recall: {train_metrics[6]:.4f}"
            )
            print(
                f"Validation Metrics - Accuracy: {val_metrics[0]:.4f}, PA: {val_metrics[1]:.4f}, Precision: {val_metrics[2]:.4f}, MeanIOU: {val_metrics[3]:.4f}, Dice: {val_metrics[4]:.4f}, F1: {val_metrics[5]:.4f}, Recall: {val_metrics[6]:.4f}"
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
