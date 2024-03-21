import os

import albumentations as A
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image

IMG_WIDTH = 1024
IMG_HEIGHT = 704
IMG_CHANNELS = 3
TRAIN_IMG_PATH = "data/train/original/"
TRAIN_MASK_PATH = "data/train/mask/"
TEST_PATH = "/test/Only_fence"


def display_image_and_mask(image, mask):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(
        image, cmap="gray"
    )  # Annahme: Grauwertbilder. Ändern Sie 'gray' zu None für Farbbilder.
    plt.title("Augmentiertes Bild")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")  # Annahme: Masken sind in Graustufen
    plt.title("Augmentierte Maske")
    plt.axis("off")

    plt.show()


def load_images_from_directory(
    directory, target_size=(1024, 704), isMask=False, threshold=30
):
    images_list = []

    # Durchlaufe alle Dateien im Verzeichnis
    for filename in os.listdir(directory):
        if filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):  # Sicherstellen, dass es ein Bildformat ist
            # Bildpfad konstruieren
            image_path = os.path.join(directory, filename)

            # Bild laden
            with Image.open(image_path) as img:
                # Bildgröße anpassen, falls gewünscht
                if isMask:
                    img = img.convert("L")

                if target_size:
                    img = img.resize(target_size)

                # Bild in ein NumPy-Array umwandeln
                img_array = np.array(img)

                if isMask:
                    img_array = (img_array > threshold).astype(np.uint8)
                    images_list.append(img_array)
                # Bild zum Ergebnis hinzufügen
                else:
                    images_list.append((img_array) / 255)

    return images_list


def conv_block_down(input_tensor, num_filters, dropout_rate, kernel_size):
    """
    Creates a convolutional block for U-Net architecture.

    Args:
    - input_tensor (tf.Tensor): Input tensor to the convolutional block.
    - num_filters (int): Number of filters for the convolutional layers.
    - dropout_rate (float): Dropout rate for regularization.
    - kernel_size (tuple): Size of the kernel for convolutional layers.

    Returns:
    - conv (tf.Tensor): Output tensor from the last convolutional layer.
    - pool (tf.Tensor): Output tensor from the max pooling layer.
    """
    # First convolutional layer
    conv = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)

    # Dropout layer
    conv = tf.keras.layers.Dropout(dropout_rate)(conv)

    # Second convolutional layer
    conv = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(conv)

    # Max pooling layer
    pool = tf.keras.layers.MaxPooling2D((2, 2))(conv)

    return conv, pool


def conv_block_up(
    input_tensor, skip_tensor, num_filters, dropout_rate, kernel_size
):
    """
    Creates a upsampling convolutional block for U-Net architecture.

    Args:
    - input_tensor (tf.Tensor): Input tensor to the convolutional block.
    - skip_tensor (tf.Tensor): Tensor for the skip connection from the downsampling path.
    - num_filters (int): Number of filters for the convolutional layers.
    - dropout_rate (float): Dropout rate for regularization.
    - kernel_size (tuple): Size of the kernel for convolutional layers.

    Returns:
    - c (tf.Tensor): Output tensor from the last convolutional layer.
    """
    # First upconvolution
    u = tf.keras.layers.Conv2DTranspose(
        num_filters,
        (2, 2),
        strides=(2, 2),
        padding="same",
    )(input_tensor)

    # Concatenating Upconvolution with Contraction tensor
    u = tf.keras.layers.concatenate([u, skip_tensor])

    # First convolutional layer
    c = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(u)

    # Dropout-Layer
    c = tf.keras.layers.Dropout(dropout_rate)(c)

    # Second convolutional layer
    c = tf.keras.layers.Conv2D(
        num_filters,
        kernel_size,
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
    )(c)

    return c


# Augmentation-Pipeline wie vorher definiert
augmentation_pipeline = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.VerticalFlip(p=0.5),
    ],
    additional_targets={"mask": "image"},
)


def augment_image_mask(image, mask):
    augmented = augmentation_pipeline(image=image, mask=mask)
    return augmented["image"], augmented["mask"]


# build the model
inputs = tf.keras.layers.Input((IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))
s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

# Contraction
c1, p1 = conv_block_down(
    input_tensor=s, num_filters=16, dropout_rate=0.1, kernel_size=(3, 3)
)
c2, p2 = conv_block_down(
    input_tensor=p1, num_filters=32, dropout_rate=0.1, kernel_size=(3, 3)
)
c3, p3 = conv_block_down(
    input_tensor=p2, num_filters=64, dropout_rate=0.2, kernel_size=(3, 3)
)
c4, p4 = conv_block_down(
    input_tensor=p3, num_filters=128, dropout_rate=0.2, kernel_size=(3, 3)
)
c5, p5 = conv_block_down(
    input_tensor=p4, num_filters=256, dropout_rate=0.2, kernel_size=(3, 3)
)
c6, p6 = conv_block_down(
    input_tensor=p5, num_filters=512, dropout_rate=0.3, kernel_size=(3, 3)
)

# Expansion
u1 = conv_block_up(
    input_tensor=c6,
    skip_tensor=c5,
    num_filters=256,
    dropout_rate=0.2,
    kernel_size=(3, 3),
)
u2 = conv_block_up(
    input_tensor=u1,
    skip_tensor=c4,
    num_filters=128,
    dropout_rate=0.2,
    kernel_size=(3, 3),
)
u3 = conv_block_up(
    input_tensor=u2,
    skip_tensor=c3,
    num_filters=64,
    dropout_rate=0.1,
    kernel_size=(3, 3),
)
u4 = conv_block_up(
    input_tensor=u3,
    skip_tensor=c2,
    num_filters=32,
    dropout_rate=0.1,
    kernel_size=(3, 3),
)
u5 = conv_block_up(
    input_tensor=u4,
    skip_tensor=c1,
    num_filters=16,
    dropout_rate=0.1,
    kernel_size=(3, 3),
)

outputs = tf.keras.layers.Conv2D(
    filters=1, kernel_size=(1, 1), activation="sigmoid"
)(u5)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)
# model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    "/artifacts/models/firstTestsModel.h5", verbose=1, save_best_only=True
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss"),
]

original_images = load_images_from_directory(
    TRAIN_IMG_PATH, (1024, 704), isMask=False
)
original_masks = load_images_from_directory(
    TRAIN_MASK_PATH, (1024, 704), isMask=True
)

augmented_images = []
augmented_masks = []

for image, mask in zip(original_images, original_masks):
    aug_image, aug_mask = augment_image_mask(image, mask)
    augmented_images.append(aug_image)
    augmented_masks.append(aug_mask)


#### TESTS #####
# for i in range(20):
# print(f"Bild {i} Shape: {augmented_images[i].shape}, Maske {i} Shape: {augmented_masks[i].shape}")
# display_image_and_mask(augmented_images[i], augmented_masks[i])


# for img_array in images:
#    print(img_array.shape)

# for img_array in masks:
#    print(img_array.shape)

# results = model.fit(
#    X, Y, validation_split=0.1, batch_size=2, epochs=5, callbacks=callbacks
# )