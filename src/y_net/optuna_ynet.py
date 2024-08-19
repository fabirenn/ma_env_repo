import os
import sys

import keras.metrics
import optuna
import tensorflow as tf
from keras.callbacks import EarlyStopping

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ynet_model import build_ynet, build_feature_extractor_for_pretraining, build_ynet_with_pretrained_semantic_extractor

from custom_callbacks import ValidationCallback
from data_loader import create_datasets_for_unet_training
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

LOG_VAL_PRED = "data/predictions/ynet"
CHECKPOINT_PATH_PRETRAINED = "artifacts/models/ynet/ynet_checkpoint_pretrained.keras"
CHECKPOINT_PATH_YNET = "artifacts/models/ynet/ynet_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

EPOCHS = 100
PATIENCE = 30


def objective(trial):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_categorical(
        "batch_size", [8, 12, 16, 20, 24, 28, 32]
    )
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.1)

    # Set the optimizer parameters
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    try:
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

        # create model & start training it
        semantic_extractor_model = build_feature_extractor_for_pretraining(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE)
        semantic_extractor_model.load_weights(CHECKPOINT_PATH_PRETRAINED)

        model = build_ynet_with_pretrained_semantic_extractor(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE, semantic_extractor_model)

        # Create the SGD optimizer
        optimizer = keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum,
            decay=weight_decay,
            nesterov=False  # You can set this to True if you want to use Nesterov momentum
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

        current_epoch = 0

        history = model.fit(
            train_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=[
                keras.callbacks.ModelCheckpoint(
                    filepath=CHECKPOINT_PATH_YNET,
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
        return val_loss


def handle_errors_during_tuning(trial, best_loss, e, current_epoch):
    print(f"The following error occured: {e}")
    trial.report(best_loss, step=current_epoch)
    raise optuna.TrialPruned()


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    study = optuna.create_study(
        direction="minimize",
        storage="sqlite:///optuna_study.db",  # Save the study in a SQLite database file
        study_name="ynet_tuning",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=200)

    # clear_directory("/work/fi263pnye-ma_data/tmp/artifacts")

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
