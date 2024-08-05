import os
import sys

import keras.metrics
from keras.models import load_model

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_loader import create_testdataset_for_unet_training
from loss_functions import dice_loss
from metrics_calculation import (
    dice_coefficient,
    f1_score,
    mean_iou,
    pixel_accuracy,
    precision,
    recall,
)
from processing import add_prediction_to_list, safe_predictions_locally

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

TEST_IMG_PATH = "data/training_test/images_mixed"
TEST_MASK_PATH = "data/training_test/labels_mixed"
CHECKPOINT_PATH = "artifacts/models/unet/unet_checkpoint.keras"
PRED_IMG_PATH = "artifacts/models/unet/pred"

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNEL = 8

BATCH_SIZE = 4

test_dataset, test_images, test_masks = create_testdataset_for_unet_training(
    directory_test_images=TEST_IMG_PATH,
    directory_test_masks=TEST_MASK_PATH,
    img_width=IMG_WIDTH,
    img_height=IMG_HEIGHT,
    batch_size=BATCH_SIZE,
    channel_size=IMG_CHANNEL,
)

model = load_model(CHECKPOINT_PATH, compile=False)
model.compile(
    optimizer="adam",
    loss=dice_loss,
    metrics=[
        "accuracy",
        pixel_accuracy,
        precision,
        mean_iou,
        dice_coefficient,
        f1_score,
        recall,
    ],
)

predictions, colored_predictions = add_prediction_to_list(
    test_dataset, model=model, batch_size=BATCH_SIZE, apply_crf=False
)

safe_predictions_locally(
    range=range(20),
    iterator=None,
    test_images=test_images,
    predictions=colored_predictions,
    test_masks=test_masks,
    pred_img_path=PRED_IMG_PATH,
    val=False,
)

os.environ["WANDB_DIR"] = "wandb/test_unet"

wandb.init(
    project="unet",
    entity="fabio-renn",
    name="test-unet",
    mode="offline",
    config={"metric": "accuracy"},
    dir=os.environ["WANDB_DIR"],
)

test_results = model.evaluate(test_images, test_masks, return_dict=True)
print(test_results)
wandb.log(test_results)
wandb.finish()
