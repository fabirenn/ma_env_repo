import os

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_CHANNELS = 3

TEST_IMG_PATH = "data/test/original/"
TEST_MASK_PATH = "data/test/mask/"
TEST_PATH = "/test/Only_fence"


def load_images_from_directory(directory):
    # puts each .png / .jpg / .jpeg File into a list of images as np-arrays and returns it

    images_list = []
    files = os.listdir(directory)
    sorted_files = sorted(files)

    # Durchlaufe alle Dateien im Verzeichnis
    for filename in sorted_files:
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Sicherstellen, dass es ein Bildformat ist
            # Bildpfad konstruieren
            image_path = os.path.join(directory, filename)

            # Bild laden
            with Image.open(image_path) as img:
                # Store Image as np-array
                img_array = np.array(img)

                # append img_array to list
                #print(filename)
                images_list.append(img_array)

    return images_list


def load_masks_from_directory(directory):
    # puts each .png / .jpg / .jpeg File into a list of images as np-arrays and returns it

    masks_list = []
    files = os.listdir(directory)
    sorted_files = sorted(files)

    # Durchlaufe alle Dateien im Verzeichnis
    for filename in sorted_files:
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Sicherstellen, dass es ein Bildformat ist
            # Bildpfad konstruieren
            masks_path = os.path.join(directory, filename)

            # Bild laden
            with Image.open(masks_path) as mask:
                mask = mask.convert("L")
                mask_array = np.array(mask)

                # append img_array to list
                print(filename)
                masks_list.append(mask_array)

    return masks_list


def resize_images(
    images_list,
    target_width,
    target_height,
    interpolation_method=None,
    scalefactor=None,
):
    # takes an image-list with np-arrays and resizes the images to target height/width
    # with interpolation_method and scalefactor, even a crop is possible
    # it returns the list with images in the correct size
    resized_images_list = []

    for img in images_list:
        resized_image = cv2.resize(
            img,
            dsize=(target_width, target_height),
            interpolation=interpolation_method,
            fx=scalefactor,
            fy=scalefactor,
        )
        resized_images_list.append(resized_image)

    return resized_images_list


def normalize_image_data(images_list):
    # take the list of np-array images and normalizes their values so that each
    # pixel has a value of minimum 0 and maximum 1
    # returns the list as normalized images_list
    normalized_images_list = []

    for img in images_list:
        normalized_images_list.append((img) / 255)

    return normalized_images_list


def make_binary_masks(mask_list, threshold):
    # the list of images containing the masks can be transformed in images
    # where pixel either have 0 as value or 1 as value
    # after transformation each masks contains of just 0 or 1 values
    # the transformed masks are returned in a list of npArrays
    binary_mask_list = []

    for mask in mask_list:
        binary_mask = np.where(mask > threshold, 1, 0)
        binary_mask_list.append(binary_mask)

    return binary_mask_list


def augment_image_mask(image, mask):
    # each image, mask pair goes through the augmentation_pipeline and
    # is then return back as pair of image & mask
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented["image"], augmented["mask"]


# augmentation pipeline, still having no tasks ==> which augmentation step make
# sense in my usecase?
augmentation_pipeline = A.Compose(
    [],
    additional_targets={"mask": "image"},
)


def convert_to_tensor(images_np_array_list, dtype=tf.float32):
    # function that converts the list of images as np-arrays into a list of tensors
    image_list_as_tensor = tf.convert_to_tensor(
        images_np_array_list, dtype=dtype
    )
    return image_list_as_tensor


def create_dataset(images_as_tensors, masks_as_tensors, batchsize, buffersize):
    # method that creates a dataset, out of images and their corresponding masks
    # furthermore, the batch and buffersize are being set
    dataset = tf.data.Dataset.from_tensor_slices(
        (images_as_tensors, masks_as_tensors)
    )
    dataset = dataset.batch(batch_size=batchsize).prefetch(
        buffer_size=buffersize
    )

    return dataset
