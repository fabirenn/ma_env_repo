import os
import sys

import keras.metrics
import optuna
import tensorflow as tf
from deeplab_model import DeepLab
from keras.callbacks import EarlyStopping
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_callbacks import ValidationCallback, dice_score, specificity_score, clear_directory
from data_loader import create_datasets_for_segnet_training
from loss_functions import dice_loss, iou_loss

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TRAIN_IMG_PATH = "data/training_train/images_mixed"
TRAIN_MASK_PATH = "data/training_train/labels_mixed"
VAL_IMG_PATH = "data/training_val/images_mixed"
VAL_MASK_PATH = "data/training_val/labels_mixed"

CHECKPOINT_PATH = "artifacts/models/deeplab/deeplab_checkpoint.h5"
LOG_VAL_PRED = "data/predictions/deeplab"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3
EPOCHS = 50
PATIENCE = 30

os.environ["WANDB_DIR"] = "wandb/train_deeplab"
os.environ["WANDB_DATA_DIR"] = "/work/fi263pnye-ma_data/tmp"


def objective(trial):
    # Hyperparameter tuning
    BATCH_SIZE = trial.suggest_categorical("batch_size", [4, 8, 12, 16])
    DROPOUT_RATE = trial.suggest_float("dropout_rate", 0.0, 0.4, step=0.1)
    loss_function = trial.suggest_categorical(
        "loss_function", ["cross_entropy", "dice_loss", "iou_loss"]
    )

    # Map the loss function name to the actual function
    loss_function_map = {
        "cross_entropy": keras.losses.binary_crossentropy,
        "dice_loss": dice_loss,
        "iou_loss": iou_loss,
    }

    #tf.keras.backend.clear_session()

    try:
        train_dataset, val_dataset = create_datasets_for_segnet_training(
            directory_train_images=TRAIN_IMG_PATH,
            directory_train_masks=TRAIN_MASK_PATH,
            directory_val_images=VAL_IMG_PATH,
            directory_val_masks=VAL_MASK_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            batch_size=BATCH_SIZE,
        )

        wandb.init(
            project="deeplab",
            entity="fabio-renn",
            mode="offline",
            name=f"train-deeplab-{trial.number}",
            config={
                "metric": "accuracy",
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
            },
            dir=os.environ["WANDB_DIR"],
        )

        model = DeepLab(
            input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
            dropout_rate=DROPOUT_RATE,
        )

        model.compile(
            optimizer="adam",
            loss=loss_function_map[loss_function],
            metrics=[
                "accuracy",
                keras.metrics.BinaryIoU(),
                keras.metrics.Precision(),
                keras.metrics.Recall(),
                specificity_score,
                dice_score,
            ],
        )

        history = model.fit(
            train_dataset,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=val_dataset,
            callbacks=[
                WandbMetricsLogger(log_freq="epoch"),
                WandbModelCheckpoint(
                    filepath=CHECKPOINT_PATH,
                    save_best_only=True,
                    save_weights_only=False,
                    monitor="val_loss",
                    verbose=1,
                ),
                ValidationCallback(
                    model=model,
                    train_data=train_dataset,
                    validation_data=val_dataset,
                    log_dir=LOG_VAL_PRED,
                    apply_crf=False,
                ),
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=PATIENCE,
                    restore_best_weights=True,
                ),
            ],
        )

        val_loss = min(history.history["val_loss"])
        wandb.finish()

        return val_loss
    except tf.errors.ResourceExhaustedError:
        print(
            "Resource exhausted error caught. GPU may not have enough memory."
        )
        return float("inf")
    except Exception as e:
        print(f"An exception occurred: {e}")
        return float("inf")


if __name__ == "__main__":
    tf.config.optimizer.set_experimental_options({"layout_optimizer": False})

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    #clear_directory("/work/fi263pnye-ma_data/tmp/artifacts")

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params:")
    for key, value in trial.params.items():
        print(f"  {key}: {value}")
