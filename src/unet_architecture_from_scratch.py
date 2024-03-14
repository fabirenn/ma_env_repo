import numpy as np
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img

IMG_WIDTH = 512
IMG_HEIGHT = 512
IMG_CHANNELS = 3


def prepare_image_and_mask(image_path, mask_path, target_size=(512, 512)):
    # Load and resize the image
    image = load_img(image_path, target_size=target_size, color_mode="rgb")
    image = img_to_array(image) / 255.0  # Normalize to [0,1]

    # Load and resize the mask
    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
    mask = img_to_array(mask) / 255.0  # Normalize to [0,1]
    mask = np.round(mask)  # Ensure mask is strictly binary

    return image, mask


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
model.summary()

checkpointer = tf.keras.callbacks.ModelCheckpoint(
    "/artifacts/models/firstTestsModel.h5", verbose=1, save_best_only=True
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=2, monitor="val_loss"),
]

results = model.fit(
    X, Y, validation_split=0.1, batch_size=2, epochs=5, callbacks=callbacks
)
