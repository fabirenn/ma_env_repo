import os
import sys

import keras.metrics
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from unet_model import unet

from custom_callbacks import ValidationCallback, clear_directory
from data_loader import load_images_for_unet_tuning, create_dataset_for_unet_tuning
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

LOG_VAL_PRED = "data/predictions/unet"
CHECKPOINT_PATH = "artifacts/models/unet/unet_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512

EPOCHS = 100
PATIENCE = 30


def objective(trial, train_images, train_masks, val_images, val_masks):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_categorical(
        "batch_size", [8, 12, 16, 20, 24, 28, 32]
    )
    IMG_CHANNEL = trial.suggest_categorical("img_channel", [3, 8])
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)
    LEARNING_RATE = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    NUM_BLOCKS = trial.suggest_int("num_blocks", 3, 6)
    KERNEL_SIZE = trial.suggest_categorical("kernel_size", [3, 5])
    OPTIMIZER = trial.suggest_categorical(
        "optimizer", ["sgd", "adagrad", "rmsprop", "adam"]
    )
    ACTIVATION = trial.suggest_categorical("activation", ["relu", "leaky_relu", "elu", "prelu"])
    USE_BATCHNORM = trial.suggest_categorical("use_batchnorm", [True, False])
    INITIALIZER = trial.suggest_categorical(
            "weight_initializer", ["he_normal", "he_uniform"]
        )
    

    if OPTIMIZER == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "adagrad":
        optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == "adam":
        optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Define the possible values for the number of filters
    filter_options = [16, 32, 64, 128, 256, 512, 1024]

    # Select the appropriate number of filters from the filter_options based on num_blocks
    filters_list = filter_options[:NUM_BLOCKS]

    try:
        current_epoch = 0
        val_loss = 1000

        train_dataset, val_dataset = create_dataset_for_unet_tuning(
            train_images,
            train_masks,
            val_images,
            val_masks,
            IMG_CHANNEL,
            BATCH_SIZE
        )

        model = unet(
            IMG_WIDTH,
            IMG_HEIGHT,
            IMG_CHANNEL,
            DROPOUT_RATE,
            filters_list,
            kernel_size=(KERNEL_SIZE, KERNEL_SIZE),
            activation=ACTIVATION,
            use_batchnorm=USE_BATCHNORM,
            initializer_function=INITIALIZER,
            training=True,
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
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=CHECKPOINT_PATH,
                    save_best_only=True,
                    save_weights_only=False,
                    monitor="val_loss",
                    verbose=1,
                ),
                ValidationCallback(
                    model=model,
                    validation_data=val_dataset,
                    log_dir=LOG_VAL_PRED,
                    apply_crf=False,
                    log_wandb=False
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=PATIENCE,
                    restore_best_weights=True,
                ),
            ],
        )
        val_loss = min(history.history["val_loss"])
        current_epoch = len(history.history["loss"])
        return val_loss
    except tf.errors.ResourceExhaustedError as e:
        handle_errors_during_tuning(trial=trial, best_loss=val_loss, e=e, current_epoch=current_epoch)
    except Exception as e:
        handle_errors_during_tuning(trial=trial, best_loss=val_loss, e=e, current_epoch=current_epoch)


def handle_errors_during_tuning(trial, best_loss, e, current_epoch):
    print(f"The following error occured: {e}")
    trial.report(best_loss, step=current_epoch)
    raise optuna.TrialPruned()


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})
    
    train_images, train_masks, val_images, val_masks = load_images_for_unet_tuning(
        directory_train_images=TRAIN_IMG_PATH,
        directory_train_masks=TRAIN_MASK_PATH,
        directory_val_images=VAL_IMG_PATH,
        directory_val_masks=VAL_MASK_PATH,
        img_width=IMG_WIDTH,
        img_height=IMG_HEIGHT,
    )

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_study.db",  # Save the study in a SQLite database file
        study_name="unet_tuning",
        load_if_exists=True,
    )
    study.optimize(lambda trial: objective(trial, train_images, train_masks, val_images, val_masks), n_trials=200)

    # clear_directory("/work/fi263pnye-ma_data/tmp/artifacts")

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
