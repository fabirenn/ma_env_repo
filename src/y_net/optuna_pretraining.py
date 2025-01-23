import os
import sys

import keras
import keras.metrics
import optuna
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ynet_model import (
    build_feature_extractor_for_pretraining,
    build_ynet,
    build_ynet_with_pretrained_semantic_extractor,
)

from data_loader import create_dataset_for_tuning, load_images_for_tuning
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

CHECKPOINT_PATH_PRETRAINED = (
    "artifacts/models/ynet/ynet_checkpoint_pretrained.keras"
)

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

EPOCHS = 100
PATIENCE = 30

STORAGE = "sqlite:///artifacts/models/ynet/optuna_ynet_pretraining.db"
STUDY_NAME = "ynet_pretraining_tuning"

TRIALS = 200


def objective(trial, train_images, train_masks, val_images, val_masks):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_int("batch_size", 4, 24, step=4)
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.5, step=0.1)

    # Set the optimizer parameters
    momentum = trial.suggest_float("momentum", 0.7, 0.99)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    optimizer = keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=momentum,
        decay=weight_decay,
        nesterov=False,
    )

    try:
        current_epoch = 0
        val_loss = 1000

        train_dataset, val_dataset = create_dataset_for_tuning(
            train_images, train_masks, val_images, val_masks, BATCH_SIZE
        )

        print("Created the datasets..")

        # create model & start training it
        semantic_extractor_model = build_feature_extractor_for_pretraining(
            IMG_WIDTH,
            IMG_HEIGHT,
            IMG_CHANNEL,
            DROPOUT_RATE,
        )
        # semantic_extractor_model.load_weights(CHECKPOINT_PATH_PRETRAINED)

        # model = build_ynet_with_pretrained_semantic_extractor(IMG_WIDTH,
        # IMG_HEIGHT, IMG_CHANNEL, DROPOUT_RATE, semantic_extractor_model)

        semantic_extractor_model.compile(
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

        history = semantic_extractor_model.fit(
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
        current_epoch = len(history.history["loss"])
        print(f"Training completed. Final Validation Loss: {val_loss}")

        trial.report(val_loss, step=current_epoch)
        print("Reported to Optuna.")

        if trial.should_prune():
            print("Trial is pruned.")
            raise optuna.TrialPruned()

        return val_loss
    except tf.errors.ResourceExhaustedError as e:
        handle_errors_during_tuning(
            trial=trial, best_loss=val_loss, e=e, current_epoch=current_epoch
        )
        return float("inf")
    finally:
        # Clear GPU memory
        keras.backend.clear_session()
        print("Cleared GPU memory after trial.")


def handle_errors_during_tuning(trial, best_loss, e, current_epoch):
    print(f"The following error occured: {e}")
    trial.report(best_loss, step=current_epoch)
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
    study.optimize(
        lambda trial: objective(
            trial, train_images, train_masks, val_images, val_masks
        ),
        n_trials=TRIALS,
    )

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
