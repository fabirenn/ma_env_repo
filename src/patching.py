import os
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import numpy.typing as npt
import torch
from dotenv import load_dotenv
from rich.progress import track

import wandb
from src.nn.models.modules.segmentation import SegmentationModule

# from src.utils.transform.mask.class2color.multiclass_numpy import (
#     class_mask2bgr_mask,
# )
from src.utils.transform.mask.change_format import mask2_2d_mask
from src.wandb.templates.mask import fill_wandb_mask_template

load_dotenv()
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", default="artifacts/")
DATA_DIR = os.getenv("DATA_DIR", default="data/")


def predict_by_patching(
    model: SegmentationModule,
    img: npt.NDArray[np.uint8],
    patch_size: tuple[int, int],  # h, w
    overlap: tuple[int, int],  # h, w,
    n_classes: int,
    transform: A.Compose = A.Compose([]),
    accumulate_in_batch: bool = True,
) -> npt.NDArray[np.float32]:
    patch_h, patch_w = patch_size
    overlap_h, overlap_w = overlap
    assert patch_h / overlap_h > 2
    assert patch_w / overlap_w > 2

    img = transform(image=img)["image"]

    if img.shape[2] < img.shape[0] and img.shape[2] < img.shape[1]:
        img = img.transpose(2, 0, 1)

    img_h, img_w = img.shape[1:]
    n_cols = 1 + int(np.ceil((img_w - patch_w) / (patch_w - overlap_w)))
    n_rows = 1 + int(np.ceil((img_h - patch_h) / (patch_h - overlap_h)))

    patches = []
    for i in range(n_rows):
        for j in range(n_cols):
            if i == n_rows - 1:
                top = img_h - patch_h
            else:
                top = i * (patch_h - overlap_h)
            if j == n_cols - 1:
                left = img_w - patch_w
            else:
                left = j * (patch_w - overlap_w)
            patch = img[:, top : top + patch_h, left : left + patch_w]
            patches += [patch]

    if accumulate_in_batch:
        batch = torch.tensor(
            np.stack(patches, axis=0),
            device=str(model.device),
        )
        with torch.no_grad():
            preds = [pred for pred in model(batch).cpu().numpy()]
    else:
        preds = []
        for patch in track(patches, "Predicting patches"):
            batch = torch.tensor(
                np.expand_dims(patch, axis=0),
                device=str(model.device),
            )
            with torch.no_grad():
                pred = model(batch).cpu().numpy()[0]
            preds += [pred]

    final_mask = np.zeros((n_classes, img_h, img_w), dtype=np.float32)

    for i in range(n_rows * n_cols):
        row = int(i / n_cols)
        col = int(i % n_cols)
        if row == n_rows - 1:
            top = img_h - patch_h
        else:
            top = row * (patch_h - overlap_h)

        if col == n_cols - 1:
            left = img_w - patch_w
        else:
            left = col * (patch_w - overlap_w)
        previous_pred = final_mask[
            :, top : top + patch_h, left : left + patch_w
        ]
        final_mask[:, top : top + patch_h, left : left + patch_w] = np.max(
            [preds[i], previous_pred], axis=0
        )
    return final_mask


if __name__ == "__main__":
    img_path = Path(DATA_DIR + "fence/IMG_0002.jpg")
    assert img_path.is_file()
    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    mean = (
        np.array((99.34108047598379, 130.66293960985723, 130.22394853555812))
        / 255
    )
    std = (
        np.array((83.42668484046382, 62.00583352483456, 62.14980049757769))
        / 255
    )
    transform = A.Compose(
        [
            A.Normalize(mean=mean, std=std, max_pixel_value=255),
            # A.Resize(3800, 5700),
            A.CenterCrop(2400, 3600),
        ]
    )

    checkpoint_path = Path(
        ARTIFACTS_DIR + "models/segment/fence/unet/2024-05-10/"
        "epoch=169-step=2040.ckpt"
    )
    segmentation_module = SegmentationModule.load_from_checkpoint(
        checkpoint_path
    )
    segmentation_module.eval()
    print("Cuda available:", torch.cuda.is_available())
    segmentation_module.cuda()
    print("Model device:", str(segmentation_module.device))
    predictions = predict_by_patching(
        model=segmentation_module,
        img=img,
        transform=transform,
        # patch_size=(600, 900),
        patch_size=(320, 480),
        overlap=(40, 60),
        n_classes=5,
        accumulate_in_batch=False,
    )
    # np.save(
    #     ARTIFACTS_DIR + f"tmp-{img_path.stem}_5184x3456.npy",
    #     pred_cls_mask)
    # cls2bgr = {
    #     1: (255, 221, 51),  # wire
    #     2: (195, 177, 241),  # post
    #     3: (49, 147, 245),  # tensioner
    #     4: (102, 255, 102),  # other
    # }
    # bgr_mask = class_mask2bgr_mask(
    #     class_mask=pred_cls_mask,
    #     cls2bgr=cls2bgr,
    # )
    preds_2d = mask2_2d_mask(predictions)

    # get wandb image
    cls2name = {
        1: "wire",
        2: "post",
        3: "tensioner",
        4: "other",
    }
    wandb_mask = fill_wandb_mask_template(
        cls2name=cls2name,
        ground_truth_2d_mask=None,
        prediction_2d_mask=preds_2d,
    )
    wandb_img = wandb.Image(
        A.center_crop(img[..., ::-1], 2400, 3600),
        masks=wandb_mask,
    )

    with wandb.init(
        job_type="patches_prediction", project="fence_segmentation"
    ) as run:
        run.log({"segmented patches": wandb_img})
    # save_dir = Path(ARTIFACTS_DIR + f"masks/patches/{img_h}x{img_w}")
    # save_dir.mkdir(exist_ok=True, parents=True)
    # cv2.imwrite(
    #     ARTIFACTS_DIR + f"masks/patches/{img_path.stem}.png",
    #     bgr_mask
    # )
