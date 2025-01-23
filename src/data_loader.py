import os

import albumentations as A
import cv2
import numpy as np
import tensorflow as tf

cls2bgr = {
    1: (255, 221, 51),  # wire
    2: (195, 177, 241),  # post
    3: (49, 147, 245),  # tensioner
    4: (102, 255, 102),  # other
}

train_class_frequencies = []
val_class_frequencies = []
test_class_frequencies = []


def count_class_pixels(masks_list, num_classes):
    """
    Count the number of pixels for each class in a list of masks.

    Args:
    masks_list (list of np.ndarray): List of masks where each mask is a one-hot
    encoded np.ndarray.
    num_classes (int): Number of classes including background.

    Returns:
    class_pixel_counts (np.ndarray): Array containing the pixel counts for each
    class.
    """
    class_pixel_counts = np.zeros(num_classes, dtype=np.int64)

    for mask in masks_list:
        # Sum the number of pixels in each class for this mask
        class_pixel_counts += np.sum(mask, axis=(0, 1)).astype(np.int64)

    return class_pixel_counts.tolist()


def bgr_mask2cls_mask(bgr_mask, cls2bgr) -> np.ndarray:
    """Convert BGR mask to class mask."""
    h, w, _ = bgr_mask.shape
    final_mask = np.zeros((h, w, len(cls2bgr) + 1), dtype=np.uint8)
    final_mask[:, :, 0] = 1  # Background

    for i, bgr in cls2bgr.items():
        bool_mask = (bgr_mask == bgr).all(-1)
        final_mask[:, :, i][bool_mask] = 1
        final_mask[:, :, 0][bool_mask] = 0

    return final_mask


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

            img = cv2.imread(image_path)
            if img is not None:
                images_list.append(img)
            else:
                print(f"Failed to read mask: {image_path}")

    return images_list


def load_masks_from_directory(directory, cls2bgr=cls2bgr):
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
            bgr_mask = cv2.imread(masks_path)
            if bgr_mask is not None:
                class_mask = bgr_mask2cls_mask(bgr_mask, cls2bgr)
                masks_list.append(class_mask)
            else:
                print(f"Failed to read mask: {masks_path}")

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


def augment_image_mask(images_list, masks_list):
    augmented_images = []
    augmented_masks = []
    # each image, mask pair goes through the augmentation_pipeline and
    # is then returned back as a pair of image & mask
    for img, mask in zip(images_list, masks_list):
        augmented = augmentation_pipeline(image=img, mask=mask)
        augmented_image = augmented["image"]
        augmented_mask = augmented["mask"]
        augmented_images.append(augmented_image)
        augmented_masks.append(augmented_mask)

    return augmented_images, augmented_masks


def preprocess_images(images_list):
    preprocess_images_list = []
    for img in images_list:
        img_uint8 = (img * 255).astype(np.uint8)
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
    [
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.5),
        A.GaussianBlur(p=0.5, sigma_limit=4, blur_limit=9),
    ],
    additional_targets={"mask": "mask"},
)


def convert_to_tensor(images_np_array_list, dtype=tf.float32):
    # function that converts the list of images as np-arrays into a list
    # of tensors
    image_list_as_tensor = tf.convert_to_tensor(
        images_np_array_list, dtype=dtype
    )
    return image_list_as_tensor


def create_dataset(images_as_tensors, masks_as_tensors, batch_size, buffersize):
    # method that creates a dataset, out of images and their corresponding masks
    # furthermore, the batch and buffersize are being set
    dataset = tf.data.Dataset.from_tensor_slices(
        (images_as_tensors, masks_as_tensors)
    )
    dataset = dataset.batch(
        batch_size=batch_size, drop_remainder=True
    ).prefetch(buffer_size=buffersize)

    return dataset


def create_datasets_for_unet_training(
    directory_train_images,
    directory_train_masks,
    directory_val_images,
    directory_val_masks,
    img_width,
    img_height,
    batch_size,
    channel_size,
):
    global train_class_frequencies
    global val_class_frequencies
    # loading img and masks from corresponding paths into to separate lists
    train_images = load_images_from_directory(directory_train_images)
    train_masks = load_masks_from_directory(directory_train_masks, cls2bgr)
    print("Train-Images successfully loaded..")

    val_images = load_images_from_directory(directory_val_images)
    val_masks = load_masks_from_directory(directory_val_masks, cls2bgr)
    print("Validation-Images successfully loaded..")

    # resizing the images to dest size for training
    train_images = resize_images(train_images, img_width, img_height)
    train_masks = resize_images(train_masks, img_width, img_height)
    val_images = resize_images(val_images, img_width, img_height)
    val_masks = resize_images(val_masks, img_width, img_height)
    print("All images resized..")

    train_class_frequencies = count_class_pixels(train_masks, 5)
    val_class_frequencies = count_class_pixels(val_masks, 5)

    print("Train class pixel counts: ", train_class_frequencies)
    print("Validation class pixel counts: ", val_class_frequencies)

    # applying augmentation to each image / mask pair
    train_images, train_masks = augment_image_mask(train_images, train_masks)
    val_images, val_masks = augment_image_mask(val_images, val_masks)

    # normalizing the values of the images and binarizing the image masks
    train_images = normalize_image_data(train_images)
    print("Train-Images normalized..")
    if channel_size > 3:
        train_images = preprocess_images(train_images)
        print("Added more channels for U-Net..")

    val_images = normalize_image_data(val_images)
    print("Val-Images normalized..")
    if channel_size > 3:
        val_images = preprocess_images(val_images)
        print("Added more channels for U-Net..")

    # converting the images/masks to tensors + expanding the masks tensor slide
    # to 1 dimension
    tensor_train_images = convert_to_tensor(train_images)
    tensor_train_masks = convert_to_tensor(train_masks)
    # tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

    tensor_val_images = convert_to_tensor(val_images)
    tensor_val_masks = convert_to_tensor(val_masks)
    # tensor_val_masks = tf.expand_dims(tensor_val_masks, axis=-1)

    print("Images converted to tensors..")

    # create dataset for training purposes
    train_dataset = create_dataset(
        tensor_train_images,
        tensor_train_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    val_dataset = create_dataset(
        tensor_val_images,
        tensor_val_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    print("Train and Validation Dataset created..")

    return train_dataset, val_dataset


def load_images_for_tuning(
    directory_train_images,
    directory_train_masks,
    directory_val_images,
    directory_val_masks,
    img_width,
    img_height,
):
    global train_class_frequencies
    global val_class_frequencies
    # loading img and masks from corresponding paths into to separate lists
    train_images = load_images_from_directory(directory_train_images)
    train_masks = load_masks_from_directory(directory_train_masks, cls2bgr)
    print("Train-Images successfully loaded..")

    val_images = load_images_from_directory(directory_val_images)
    val_masks = load_masks_from_directory(directory_val_masks, cls2bgr)
    print("Validation-Images successfully loaded..")

    # resizing the images to dest size for training
    train_images = resize_images(train_images, img_width, img_height)
    train_masks = resize_images(train_masks, img_width, img_height)
    val_images = resize_images(val_images, img_width, img_height)
    val_masks = resize_images(val_masks, img_width, img_height)
    print("All images resized..")

    train_class_frequencies = count_class_pixels(train_masks, 5)
    val_class_frequencies = count_class_pixels(val_masks, 5)

    print("Train class pixel counts: ", train_class_frequencies)
    print("Validation class pixel counts: ", val_class_frequencies)

    # applying augmentation to each image / mask pair
    train_images, train_masks = augment_image_mask(train_images, train_masks)
    val_images, val_masks = augment_image_mask(val_images, val_masks)

    return train_images, train_masks, val_images, val_masks


def create_dataset_for_unet_tuning(
    train_images, train_masks, val_images, val_masks, channel_size, batch_size
):
    # normalizing the values of the images and binarizing the image masks
    train_images = normalize_image_data(train_images)
    print("Train-Images normalized..")
    if channel_size > 3:
        train_images = preprocess_images(train_images)
        print("Added more channels for U-Net..")

    val_images = normalize_image_data(val_images)
    print("Val-Images normalized..")
    if channel_size > 3:
        val_images = preprocess_images(val_images)
        print("Added more channels for U-Net..")

    # converting the images/masks to tensors + expanding the masks tensor slide
    # to 1 dimension
    tensor_train_images = convert_to_tensor(train_images)
    tensor_train_masks = convert_to_tensor(train_masks)
    # tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

    tensor_val_images = convert_to_tensor(val_images)
    tensor_val_masks = convert_to_tensor(val_masks)
    # tensor_val_masks = tf.expand_dims(tensor_val_masks, axis=-1)

    print("Images converted to tensors..")

    # create dataset for training purposes
    train_dataset = create_dataset(
        tensor_train_images,
        tensor_train_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    val_dataset = create_dataset(
        tensor_val_images,
        tensor_val_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    print("Train and Validation Dataset created..")

    return train_dataset, val_dataset


def create_dataset_for_tuning(
    train_images, train_masks, val_images, val_masks, batch_size
):
    # normalizing the values of the images and binarizing the image masks
    train_images = normalize_image_data(train_images)
    print("Train-Images normalized..")

    val_images = normalize_image_data(val_images)
    print("Val-Images normalized..")

    # converting the images/masks to tensors + expanding the masks tensor slide
    # to 1 dimension
    tensor_train_images = convert_to_tensor(train_images)
    tensor_train_masks = convert_to_tensor(train_masks)
    # tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

    tensor_val_images = convert_to_tensor(val_images)
    tensor_val_masks = convert_to_tensor(val_masks)
    # tensor_val_masks = tf.expand_dims(tensor_val_masks, axis=-1)

    print("Images converted to tensors..")

    # create dataset for training purposes
    train_dataset = create_dataset(
        tensor_train_images,
        tensor_train_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    val_dataset = create_dataset(
        tensor_val_images,
        tensor_val_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    print("Train and Validation Dataset created..")

    return train_dataset, val_dataset


def create_datasets_for_segnet_training(
    directory_train_images,
    directory_train_masks,
    directory_val_images,
    directory_val_masks,
    img_width,
    img_height,
    batch_size,
):
    global train_class_frequencies
    global val_class_frequencies
    # loading images and masks from corresponding paths into to separate lists
    train_images = load_images_from_directory(directory_train_images)
    train_masks = load_masks_from_directory(directory_train_masks, cls2bgr)
    print("Train-Images successfully loaded..")

    val_images = load_images_from_directory(directory_val_images)
    val_masks = load_masks_from_directory(directory_val_masks, cls2bgr)
    print("Validation-Images successfully loaded..")

    # resizing the images to dest size for training
    train_images = resize_images(train_images, img_width, img_height)
    train_masks = resize_images(train_masks, img_width, img_height)
    val_images = resize_images(val_images, img_width, img_height)
    val_masks = resize_images(val_masks, img_width, img_height)
    print("All images resized..")

    train_class_frequencies = count_class_pixels(train_masks, 5)
    val_class_frequencies = count_class_pixels(val_masks, 5)

    print("Train class pixel counts: ", train_class_frequencies)
    print("Validation class pixel counts: ", val_class_frequencies)

    # applying augmentation to each image / mask pair
    train_images, train_masks = augment_image_mask(train_images, train_masks)
    val_images, val_masks = augment_image_mask(val_images, val_masks)

    # normalizing the values of the images and binarizing the image masks
    train_images = normalize_image_data(train_images)
    print("Train-Images normalized..")

    val_images = normalize_image_data(val_images)
    print("Val-Images normalized..")

    # converting the images/masks to tensors + expanding the masks tensor slide
    # to 1 dimension
    tensor_train_images = convert_to_tensor(train_images)
    tensor_train_masks = convert_to_tensor(train_masks)
    # tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

    tensor_val_images = convert_to_tensor(val_images)
    tensor_val_masks = convert_to_tensor(val_masks)
    # tensor_val_masks = tf.expand_dims(tensor_val_masks, axis=-1)

    print("Images converted to tensors..")

    # create dataset for training purposes
    train_dataset = create_dataset(
        tensor_train_images,
        tensor_train_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    val_dataset = create_dataset(
        tensor_val_images,
        tensor_val_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    print("Train and Validation Dataset created..")

    return train_dataset, val_dataset


def create_testdataset_for_unet_training(
    directory_test_images,
    directory_test_masks,
    img_width,
    img_height,
    batch_size,
    channel_size,
):
    global test_class_frequencies
    # loading images and masks from corresponding paths into to separate lists
    test_images = load_images_from_directory(directory_test_images)
    test_masks = load_masks_from_directory(directory_test_masks, cls2bgr)
    print("Test-Images successfully loaded..")

    # resizing the images to dest size for training
    test_images = resize_images(test_images, img_width, img_height)
    test_masks = resize_images(test_masks, img_width, img_height)
    print("Test-Images resized..")

    test_class_frequencies = count_class_pixels(test_masks, 5)
    print("Test class pixel counts: ", test_class_frequencies)

    # normalizing the values of the images and binarizing the image masks
    test_images_normalized = normalize_image_data(test_images)
    print("Train-Images normalized..")
    if channel_size > 3:
        test_images_normalized = preprocess_images(test_images_normalized)
        print("Added more Channels for U-Net..")

    # converting the images/masks to tensors + expanding the masks tensor slide
    # to 1 dimension
    tensor_test_images = convert_to_tensor(test_images_normalized)
    tensor_test_masks = convert_to_tensor(test_masks)
    # tensor_test_masks = tf.expand_dims(tensor_test_masks, axis=-1)
    print("Images converted to tensors..")

    # create dataset for training purposes
    test_dataset = create_dataset(
        tensor_test_images,
        tensor_test_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    print("Test Dataset created..")

    return test_dataset, test_images, test_masks


def create_testdataset_for_segnet_training(
    directory_test_images,
    directory_test_masks,
    img_width,
    img_height,
    batch_size,
):
    global test_class_frequencies
    # loading images and masks from corresponding paths into to separate lists
    test_images = load_images_from_directory(directory_test_images)
    test_masks = load_masks_from_directory(directory_test_masks, cls2bgr)
    print("Test-Images successfully loaded..")

    # resizing the images to dest size for training
    test_images = resize_images(test_images, img_width, img_height)
    test_masks = resize_images(test_masks, img_width, img_height)
    print("Test-Images resized..")

    test_class_frequencies = count_class_pixels(test_masks, 5)
    print("Test class pixel counts: ", test_class_frequencies)

    # normalizing the values of the images and binarizing the image masks
    test_images = normalize_image_data(test_images)
    print("Train-Images normalized..")
    # converting the images/masks to tensors + expanding the masks tensor slide
    # to 1 dimension
    tensor_test_images = convert_to_tensor(test_images)
    tensor_test_masks = convert_to_tensor(test_masks)
    # tensor_test_masks = tf.expand_dims(tensor_test_masks, axis=-1)
    print("Images converted to tensors..")

    # create dataset for training purposes
    test_dataset = create_dataset(
        tensor_test_images,
        tensor_test_masks,
        batch_size=batch_size,
        buffersize=tf.data.AUTOTUNE,
    )

    print("Test Dataset created..")

    return test_dataset, test_images, test_masks


def create_dataset_for_image_segmentation(img_dir, mask_dir):
    global test_class_frequencies
    images = []
    preprocessed_images = []
    images = load_images_from_directory(img_dir)
    images = resize_images(images, 1500, 1000)
    images = normalize_image_data(images)
    preprocessed_images = preprocess_images(images)

    masks = []
    masks = load_masks_from_directory(mask_dir, cls2bgr)
    masks = resize_images(masks, 1500, 1000)
    test_class_frequencies = count_class_pixels(masks, 5)
    # masks = make_binary_masks(masks, threshold=30)

    return images, preprocessed_images, masks


def create_dataset_for_mask_prediction(img_dir):
    global test_class_frequencies
    images = []
    preprocessed_images = []
    images = load_images_from_directory(img_dir)
    images = resize_images(images, 1500, 1000)
    images = normalize_image_data(images)
    preprocessed_images = preprocess_images(images)

    return images, preprocessed_images


def create_dataset_for_mask_prediction(img_dir):
    images = []
    images = load_images_from_directory(img_dir)
    images = normalize_image_data(images)
    return images
