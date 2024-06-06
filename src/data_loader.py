import os

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


def load_images_from_directory(directory):
    # puts each .png / .jpg / .jpeg File into a list of images as np-arrays
    # and returns it

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
                # print(filename)
                images_list.append(img_array)

    return images_list


def load_masks_from_directory(directory):
    # puts each .png / .jpg / .jpeg File into a list of images as np-arrays
    # and returns it

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
                # print(filename)
                masks_list.append(mask_array)

    return masks_list


def resize_images(
    images_list,
    target_width,
    target_height,
    interpolation_method=None,
    scalefactor=None,
):
    # takes an image-list with np-arrays and resizes the images to target
    # height/width
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


def preprocess_images(images_list):
    preprocess_images_list = []
    for img in images_list:
        img_uint8 = (img * 255).astype(np.uint8)
        img_cv32f = img.astype(np.float32)
        y_channel = (
            cv2.cvtColor(img_uint8, cv2.COLOR_BGR2YUV)[:, :, 0].astype(
                np.float32
            )
            / 255.0
        )

        gray_image = (
            cv2.cvtColor(img_uint8, cv2.COLOR_BGR2GRAY).astype(np.float32)
            / 255.0
        )

        # Apply Sobel filter in X and Y directions
        sobel_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)

        # Apply Laplacian filter
        laplacian_4 = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=1)
        laplacian_8 = cv2.Laplacian(gray_image, cv2.CV_32F, ksize=3)

        # Normalize Sobel and Laplacian filters
        sobel_x = cv2.normalize(sobel_x, None, 0, 1, cv2.NORM_MINMAX)
        sobel_y = cv2.normalize(sobel_y, None, 0, 1, cv2.NORM_MINMAX)
        laplacian_4 = cv2.normalize(laplacian_4, None, 0, 1, cv2.NORM_MINMAX)
        laplacian_8 = cv2.normalize(laplacian_8, None, 0, 1, cv2.NORM_MINMAX)

        # Stack the original image with the new channels
        y_channel = np.expand_dims(y_channel, axis=-1)
        sobel_x = np.expand_dims(sobel_x, axis=-1)
        sobel_y = np.expand_dims(sobel_y, axis=-1)
        laplacian_4 = np.expand_dims(laplacian_4, axis=-1)
        laplacian_8 = np.expand_dims(laplacian_8, axis=-1)

        combined = np.concatenate(
            (img, y_channel, sobel_x, sobel_y, laplacian_4, laplacian_8),
            axis=-1,
        )
        # print(combined.shape)
        preprocess_images_list.append(combined)

    return preprocess_images_list


# augmentation pipeline, still having no tasks ==> which augmentation step make
# sense in my usecase?
augmentation_pipeline = A.Compose(
    [],
    additional_targets={"mask": "image"},
)


def convert_to_tensor(images_np_array_list, dtype=tf.float32):
    # function that converts the list of images as np-arrays into a list
    # of tensors
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
