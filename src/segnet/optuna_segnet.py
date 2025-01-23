import ast
import gc
import os
import sys

import keras.metrics
import optuna
import tensorflow as tf
from numba import cuda
from segnet_model import segnet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import create_dataset_for_unet_tuning, load_images_for_tuning
from loss_functions import dice_loss
from metrics_calculation import (
    dice_coefficient,
    mean_iou,
    pixel_accuracy,
    precision,
    recall,
)

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

EPOCHS = 100
PATIENCE = 30

STORAGE = "sqlite:///artifacts/models/segnet/optuna_segnet.db"
STUDY_NAME = "segnet_tuning"

TRIALS = 200


def objective(trial, train_images, train_masks, val_images, val_masks):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_int(
        "batch_size", 4, 16, step=4
    )
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.3, step=0.1)
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-3, log=True)
    NUM_FILTERS = trial.suggest_categorical(
        "num_filters_index",
        [   
            "[16, 32, 64, 128]",
            "[32, 64, 128, 256]",
            "[64, 128, 256, 512]",
            "[128, 256, 512, 1024]",
            "[16, 32, 64, 128, 256]",
            "[32, 64, 128, 256, 512]",
            "[64, 128, 256, 512, 1024]",
            "[16, 32, 64, 128, 256, 512]",  # Fix: Added comma at the end of the previous line
            "[32, 64, 128, 256, 512, 1024]"
        ]
    )
    KERNEL_SIZE = trial.suggest_categorical("kernel_size", [3, 5])
    ACTIVATION = trial.suggest_categorical("activation", ["elu", "prelu"])
    INITIALIZER = trial.suggest_categorical(
            "weight_initializer", ["he_normal", "he_uniform"]
        )
    USE_BATCHNORM = True
    num_filters = ast.literal_eval(NUM_FILTERS)
    
    optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE, clipnorm=1.0)
    

    try:
        train_dataset, val_dataset = create_dataset_for_unet_tuning(
            train_images,
            train_masks,
            val_images,
            val_masks,
            IMG_CHANNEL,
            BATCH_SIZE
        )

        print("Created the datasets..")

        model = segnet(
            input_size=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
            dropout_rate=DROPOUT_RATE,
            num_filters=num_filters,
            kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
            activation=ACTIVATION,
            use_batchnorm=USE_BATCHNORM,
            initializer_function=INITIALIZER
        )

        model.compile(
            optimizer=optimizer,
            loss=dice_loss,
            metrics=[
                "accuracy",
                pixel_accuracy,
                precision,
                mean_iou,
                dice_coefficient,
                recall,
            ],
        )
        print("Starting training...")
        history = model.fit(
            train_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            verbose=1,
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=PATIENCE,
                ),
            ],
        )

        val_loss = min(history.history["val_loss"])
        return val_loss
    except tf.errors.ResourceExhaustedError as e:
        handle_errors_during_tuning(e)
    except Exception as e:
        handle_errors_during_tuning(e)
    finally:
        # Clear GPU memory
        keras.backend.clear_session()
        print("Cleared GPU memory after trial.")


def handle_errors_during_tuning(e):
    print(f"The following error occured: {e}")
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
        storage=STORAGE,  # Save the study in a SQLite database file
        study_name=STUDY_NAME,
        load_if_exists=False,
    )

    study.optimize(lambda trial: objective(trial, train_images, train_masks, val_images, val_masks), n_trials=TRIALS)

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
