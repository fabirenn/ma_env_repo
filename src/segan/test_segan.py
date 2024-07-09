import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
import keras.metrics
from keras.models import load_model
from keras.utils import array_to_img, img_to_array
from PIL import Image as im

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from custom_callbacks import ValidationCallback, specificity_score, dice_score
from data_loader import (
    create_testdataset_for_segnet_training,
    create_testdataset_for_unet_training,
    make_binary_masks,
)
from processing import (
    add_prediction_to_list,
    calculate_binary_dice,
    calculate_binary_iou,
    safe_predictions_locally,
)

from loss_functions import combined_loss

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TEST_IMG_PATH = "data/training_test/images_mixed"
TEST_MASK_PATH = "data/training_test/labels_mixed"

'''TEST_IMG_PATH = "data/local/test/images"
TEST_MASK_PATH = "data/local/test/labels"'''

CHECKPOINT_PATH = "artifacts/models/segan/segan_checkpoint.h5"
PRED_IMG_PATH = "artifacts/models/segan/pred"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

UNET = True

BATCH_SIZE = 4
EPOCHS = 50


if UNET is True:
    test_dataset, test_images, test_masks = (
        create_testdataset_for_unet_training(
            directory_test_images=TEST_IMG_PATH,
            directory_test_masks=TEST_MASK_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            batch_size=BATCH_SIZE,
        )
    )

else:
    test_dataset, test_images, test_masks = (
        create_testdataset_for_segnet_training(
            directory_test_images=TEST_IMG_PATH,
            directory_test_masks=TEST_MASK_PATH,
            img_width=IMG_WIDTH,
            img_height=IMG_HEIGHT,
            batch_size=BATCH_SIZE,
        )
    )


model = load_model(CHECKPOINT_PATH, compile=False)

model.compile(
    optimizer="adam",
    loss=combined_loss,
    metrics=[
        "accuracy",
        keras.metrics.BinaryIoU(),
        keras.metrics.Precision(),
        keras.metrics.Recall(),
        specificity_score,
        dice_score,
    ],
)

predictions, binary_predictions = add_prediction_to_list(
    test_dataset, model=model, batch_size=BATCH_SIZE, apply_crf=False
)

safe_predictions_locally(
    range=range(50),
    iterator=None,
    test_images=test_images,
    predictions=binary_predictions,
    test_masks=test_masks,
    pred_img_path=PRED_IMG_PATH,
    val=False,
)
os.environ["WANDB_DIR"] = "wandb/testing_models"

wandb.init(
    project="image-segmentation",
    entity="fabio-renn",
    name="test-segan",
    mode="offline",
    config={"metric": "accuracy"},
    dir=os.environ["WANDB_DIR"],
)

test_results = model.evaluate(test_images, test_masks, return_dict=True)
print(test_results)
wandb.log(test_results)
wandb.finish()
