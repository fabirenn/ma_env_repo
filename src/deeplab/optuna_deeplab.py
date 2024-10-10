import ast
import os
import sys
import gc
from numba import cuda

import keras.metrics
import optuna
import tensorflow as tf
from deeplab_model import DeepLab

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import load_images_for_tuning, create_dataset_for_unet_tuning
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


def objective(trial, train_images, train_masks, val_images, val_masks):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_int(
        "batch_size", 4, 24, step=4
    )
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
    FILTERS = trial.suggest_categorical("filters", [64, 128, 256, 512, 1024])
    KERNEL_SIZE = trial.suggest_categorical("kernel_size", [3, 5])
    USE_BATCHNORM = trial.suggest_categorical("use_batchnorm", [True, False])
    ACTIVATION = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "prelu"])
    INITIALIZER = trial.suggest_categorical(
            "weight_initializer", ["he_normal", "he_uniform"]
        )
    OPTIMIZER = trial.suggest_categorical(
        "optimizer", ["sgd", "adagrad", "rmsprop", "adam"]
    )
    DILATION_RATES = trial.suggest_categorical(
        "dilation_rates",
        [
            "[1, 2, 4]",
            "[2, 4, 8]",
            "[2, 5, 7]",
            "[3, 6, 9]",
            "[4, 8, 16]",
            "[3, 9, 27]",
            "[6, 12, 18]",
            "[4, 10, 20]",
            "[6, 18, 36]",
            "[8, 16, 32]",
            "[12, 24, 36]",
            "[8, 20, 40]",
            "[1, 2, 4, 8]",
            "[2, 4, 8, 16]",
            "[3, 6, 9, 18]",
        ],
    )

    if OPTIMIZER == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    try:
        train_dataset, val_dataset = create_dataset_for_unet_tuning(
            train_images,
            train_masks,
            val_images,
            val_masks,
            IMG_CHANNEL,
            BATCH_SIZE,
        )

        print("Created the datasets..")

        dilation_rates = ast.literal_eval(DILATION_RATES)

        model = DeepLab(
            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
            dropout_rate=DROPOUT_RATE,
            filters=FILTERS,
            dilation_rates=dilation_rates,
            use_batchnorm=USE_BATCHNORM,
            kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
            initializer_function=INITIALIZER,
            activation=ACTIVATION
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
        handle_errors_during_tuning(trial, e)
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
        storage="sqlite:///optuna_deeplab.db",  # Save the study in a SQLite database file
        study_name="deeplab_tuning",
        load_if_exists=True,
    )
    
    study.optimize(lambda trial: objective(trial, train_images, train_masks, val_images, val_masks), n_trials=200)

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
