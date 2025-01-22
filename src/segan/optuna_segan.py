import os
import sys

import keras
import numpy as np
import traceback
import optuna
import segmentation_models as sm
import tensorflow as tf
from segan_model import discriminator

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "u_net"))
)
from unet_model import unet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_callbacks import ValidationCallback
from data_loader import create_dataset_for_unet_tuning, load_images_for_tuning
from loss_functions import (
    combined_discriminator_loss,
    combined_generator_loss,
    discriminator_loss,
    generator_loss,
    dice_loss
)
from metrics_calculation import (
    accuracy,
    dice_coefficient,
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

IMG_WIDTH = 512
IMG_HEIGHT = 512

EPOCHS = 100
PATIENCE = 30
BEST_IOU = 0
WAIT = 0


def objective(trial, train_images, train_masks, val_images, val_masks):
    print("Starting objective function...") 
    BATCH_SIZE = trial.suggest_int(
        "batch_size", 12, 24, step=4
    )
    IMG_CHANNEL = 3
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.1, step=0.1)
    GENERATOR_TRAINING_STEPS = trial.suggest_int("g_training_steps", 6, 10)
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-2, 1e-1, log=True)
    FILTERS_DEPTH = 6
    KERNEL_SIZE = trial.suggest_categorical("kernel_size", [3, 5])
    ACTIVATION = trial.suggest_categorical("activation", ["leaky_relu", "elu"])
    
    optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)

    filters_list = [16, 32, 64, 128, 256, 512, 1024]  # Base list of filters
    discriminator_filters = filters_list[:FILTERS_DEPTH]

    #current_epoch = 0

    train_dataset, val_dataset = create_dataset_for_unet_tuning(
        train_images,
        train_masks,
        val_images,
        val_masks,
        IMG_CHANNEL,
        BATCH_SIZE
    )
    print("Created the datasets..")

    generator_model = unet(
        IMG_WIDTH,
        IMG_HEIGHT,
        IMG_CHANNEL,
        DROPOUT_RATE,
        discriminator_filters,
        kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
        activation=ACTIVATION,
        use_batchnorm=True,
        initializer_function="he_uniform",
        training=True,
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

    gen_optimizer = optimizer
    disc_optimizer = optimizer

    #generator_model.summary()
    #discriminator_model.summary()

    generator_model.compile(
        optimizer=gen_optimizer, loss=generator_loss, metrics=["accuracy"]
    )

    discriminator_model.compile(
        optimizer=disc_optimizer, loss=discriminator_loss, metrics=["accuracy"]
    )

    def evaluate_generator(generator, dataset):
        metrics = {
            "dice": keras.metrics.Mean(name="dice"),
            "mean_iou": keras.metrics.Mean(name="mean_iou"),
            "pixel_accuracy": keras.metrics.Mean(name="pixel_accuracy"),
            "precision": keras.metrics.Mean(name="precision"),
            "recall": keras.metrics.Mean(name="recall"),
        }
        val_loss = 0
        total_batches = 0

        # Calculate metrics over the validation dataset
        for image_batch, mask_batch in dataset:
            predictions = generator(image_batch, training=False)

            # Segmentation loss
            dice_loss_value = dice_loss(mask_batch, predictions)

            val_loss += dice_loss_value.numpy()

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
            total_batches += 1
        
        val_loss /= total_batches
        results = {
            "val_loss": val_loss,
            "dice": metrics["dice"].result().numpy(),
            "mean_iou": metrics["mean_iou"].result().numpy(),
            "pixel_accuracy": metrics["pixel_accuracy"].result().numpy(),
            "precision": metrics["precision"].result().numpy(),
            "recall": metrics["recall"].result().numpy(),
        }
        return results

    @tf.function
    def train_step_generator(images, masks):
        with tf.GradientTape() as gen_tape:
            generated_masks = generator_model(images, training=True)
            gen_loss, segmentation_loss = combined_generator_loss(discriminator_model, intermediate_layer_model, images, masks, generated_masks)
        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator_model.trainable_variables
        )
        gen_optimizer.apply_gradients(
            zip(gradients_of_generator, generator_model.trainable_variables)
        )
        return segmentation_loss

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
        best_val_loss = float("inf")
        try:
            for epoch in range(epochs):
                print(f"Epoch {epoch+1}/{EPOCHS}")
                for image_batch, mask_batch in train_dataset:
                    for _ in range(trainingsteps):
                        gen_loss = train_step_generator(image_batch, mask_batch)
                    disc_loss = train_step_discriminator(image_batch, mask_batch)
                val_metrics = evaluate_generator(generator_model, val_dataset)
                val_loss = val_metrics["val_loss"]
                print(
                    f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f} - Validation Loss: {val_loss:.4f}"
                )
                print(
                    f"Validation Metrics - PA: {val_metrics['pixel_accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, IOU: {val_metrics['mean_iou']:.4f}, Dice: {val_metrics['dice']:.4f}"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    WAIT = 0
                else:
                    WAIT += 1
                    if WAIT >= PATIENCE:
                        print("Early stopping triggered\n")
                        return best_val_loss
            
            print(f"Training completed. Final Validation Loss: {val_loss}")
            return best_val_loss
            
        except tf.errors.ResourceExhaustedError as e:
            handle_errors_during_tuning(e)
        except Exception as e:
            handle_errors_during_tuning(e)
        finally:
            # Clear GPU memory
            keras.backend.clear_session()
            print("Cleared GPU memory after trial.")

    best_gen_loss = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=EPOCHS,
        trainingsteps=GENERATOR_TRAINING_STEPS,
    )

    return best_gen_loss


def handle_errors_during_tuning(e):
    print(f"The following error occured: {e}")
    traceback.print_exc() 
    raise optuna.TrialPruned()


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    
    print("Going to load the data...")
    train_images, train_masks, val_images, val_masks = load_images_for_tuning(
        directory_train_images=TRAIN_IMG_PATH,
        directory_train_masks=TRAIN_MASK_PATH,
        directory_val_images=VAL_IMG_PATH,
        directory_val_masks=VAL_MASK_PATH,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
    )
    print("Loaded Images, now starting with the study")

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_segan_simple.db",  # Save the study in a SQLite database file
        study_name="segan_tuning",
        load_if_exists=False,
    )

    study.optimize(lambda trial: objective(trial, train_images, train_masks, val_images, val_masks), n_trials=50)

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")