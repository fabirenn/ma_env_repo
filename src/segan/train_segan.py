import os
import sys

import keras
import keras.backend
import numpy as np
import segmentation_models as sm
import tensorflow as tf
from segan_model import discriminator

import wandb

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "u_net"))
)
from unet_model import unet

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import create_datasets_for_unet_training
from loss_functions import (
    combined_discriminator_loss,
    combined_generator_loss,
    dice_loss,
    discriminator_loss,
    generator_loss,
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

'''
TRAIN_IMG_PATH = "data/local/train/images"
TRAIN_MASK_PATH = "data/local/train/labels"
VAL_IMG_PATH = "data/local/val/images"
VAL_MASK_PATH = "data/local/val/labels"'''


LOG_VAL_PRED = "data/predictions/segan"
CHECKPOINT_PATH = "./artifacts/models/segan/segan_checkpoint.keras"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 3

DROPOUT_RATE = 0.0
LEARNING_RATE = 0.02446071789
BATCH_SIZE = 16
FILTERS_LIST = [16, 32, 64, 128, 256, 512]
KERNELSIZE = 5
ACTIVATION = "elu"
INITIALIZER = "he_uniform"
USE_BATCHNORM = True


GENERATOR_TRAINING_STEPS = 9

EPOCHS = 200
PATIENCE = 70
BEST_IOU = 0
WAIT = 0


os.environ["WANDB_DIR"] = "wandb/train_segan"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Start a run, tracking hyperparameters
wandb.init(
    # set the wandb project where this run will be logged
    project="segan",
    entity="fabio-renn",
    mode="offline",
    name="train-segan",
    # track hyperparameters and run metadata with wandb.config
    config={"metric": "accuracy", "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    dir=os.environ["WANDB_DIR"],
)

# [optional] use wandb.config as your config
config = wandb.config

#keras.backend.set_image_data_format("channels_last")


generator_model = unet(
    IMG_WIDTH,
    IMG_HEIGHT,
    IMG_CHANNEL,
    DROPOUT_RATE,
    FILTERS_LIST,
    kernel_size=(KERNELSIZE, KERNELSIZE),
    activation=ACTIVATION,
    use_batchnorm=USE_BATCHNORM,
    initializer_function=INITIALIZER,
    training=True,

)


discriminator_model = discriminator(
    (IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL),
    (IMG_WIDTH, IMG_HEIGHT, 5),
    FILTERS_LIST,
)

# Create the intermediate model
intermediate_layer_model = keras.Model(
    inputs=discriminator_model.input,
    outputs=[layer.output for layer in discriminator_model.layers if 'conv' in layer.name or 'bn' in layer.name]
)

# loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
# loss_fn = combined_loss
gen_optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
disc_optimizer = keras.optimizers.Adagrad(learning_rate=LEARNING_RATE)
checkpoint = tf.train.Checkpoint(
    generator_optimizer=gen_optimizer,
    discriminator_optimizer=disc_optimizer,
    generator=generator_model,
    discriminator=discriminator_model,
)

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
    }
    
    val_loss = 0
    total_batches = 0

    # Calculate metrics over the validation dataset
    for image_batch, mask_batch in dataset:
        predictions = generator(image_batch, training=False)

        # Segmentation loss
        #cce = keras.losses.CategoricalCrossentropy(from_logits=False)
        #segmentation_loss = cce(mask_batch, predictions)

        #val_loss += segmentation_loss.numpy()
	#cce = keras.losses.CategoricalCrossentropy(from_logits=False)
        #segmentation_loss = cce(mask_batch, predictions)
        dice_loss_value = dice_loss(mask_batch, predictions)

        val_loss += dice_loss_value.numpy()
        metrics["accuracy"].update_state(accuracy(mask_batch, predictions))
        metrics["dice"].update_state(dice_coefficient(mask_batch, predictions))
        metrics["mean_iou"].update_state(mean_iou(mask_batch, predictions))
        metrics["pixel_accuracy"].update_state(
            pixel_accuracy(mask_batch, predictions)
        )
        metrics["precision"].update_state(precision(mask_batch, predictions))
        metrics["recall"].update_state(recall(mask_batch, predictions))
        total_batches += 1

    val_loss /= total_batches
    results = {
        "val_loss": val_loss,
        "accuracy": metrics["accuracy"].result().numpy(),
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
    # gradients_of_generator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_generator]
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
    # gradients_of_discriminator = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients_of_discriminator]
    disc_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator_model.trainable_variables)
    )
    return disc_loss


def generate_images(model, dataset, epoch):
    sample = dataset.take(1)
    image_batch, mask_batch = next(iter(sample))
    pred_batch = model.predict(image_batch)
    x = image_batch[0]
    x_rgb = x[..., :3][..., ::-1]

    x_rgb = x_rgb.numpy() if isinstance(x_rgb, tf.Tensor) else x_rgb

    safe_predictions_locally(
        range=None,
        iterator=epoch,
        test_images=x_rgb,
        predictions=pred_batch[0],
        test_masks=mask_batch[0],
        pred_img_path=LOG_VAL_PRED,
        val=True,
    )


def train(train_dataset, val_dataset, epochs, trainingsteps):
    global WAIT
    best_val_loss = float("inf")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{EPOCHS}")

        for image_batch, mask_batch in train_dataset:
            for _ in range(trainingsteps):
                gen_loss = train_step_generator(image_batch, mask_batch)
                # clip_discriminator_weights(discriminator_model, clip_value)
            disc_loss = train_step_discriminator(image_batch, mask_batch)

        train_metrics = evaluate_generator(generator_model, train_dataset)
        val_metrics = evaluate_generator(generator_model, val_dataset)
        val_loss = val_metrics["val_loss"]
        wandb.log(
            {
                "epoch": epoch + 1,
                "gen_loss": gen_loss,
                "disc_loss": disc_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_pixel_accuracy": train_metrics["pixel_accuracy"],
                "train_precision": train_metrics["precision"],
                "train_mean_iou": train_metrics["mean_iou"],
                "train_dice_coefficient": train_metrics["dice"],
                "train_recall": train_metrics["recall"],
                "val_accuracy": val_metrics["accuracy"],
                "val_pixel_accuracy": val_metrics["pixel_accuracy"],
                "val_precision": val_metrics["precision"],
                "val_mean_iou": val_metrics["mean_iou"],
                "val_dice_coefficient": val_metrics["dice"],
                "val_recall": val_metrics["recall"],
            }
        )

        print(
            f"Generator Loss: {gen_loss:.4f} - Discriminator Loss: {disc_loss:.4f} - Validation Loss: {val_loss:.4f}"
        )
        print(
            f"Train Metrics - Accuracy: {train_metrics['accuracy']:.4f}, PA: {train_metrics['pixel_accuracy']:.4f}, Precision: {train_metrics['precision']:.4f}, Recall: {train_metrics['recall']:.4f}, IOU: {train_metrics['mean_iou']:.4f}, Dice: {train_metrics['dice']:.4f}"
        )
        print(
            f"Validation Metrics - Accuracy: {val_metrics['accuracy']:.4f}, PA: {val_metrics['pixel_accuracy']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}, IOU: {val_metrics['mean_iou']:.4f}, Dice: {val_metrics['dice']:.4f}"
        )

        generate_images(model=generator_model, dataset=val_dataset, epoch=epoch)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint.save(file_prefix=CHECKPOINT_PATH)
            generator_model.save(CHECKPOINT_PATH)
            print("Improved & Saved model\n")
            WAIT = 0
        else:
            WAIT += 1
            if WAIT >= PATIENCE:
                print("Early stopping triggered\n")
                return
            else:
                print("Waiting for improvement..\n")


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


train(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    epochs=EPOCHS,
    trainingsteps=GENERATOR_TRAINING_STEPS,
)

wandb.finish()
