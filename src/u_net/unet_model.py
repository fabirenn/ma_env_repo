import tensorflow as tf
import keras
from keras.layers import BatchNormalization


def unet(
    img_width,
    img_height,
    img_channels,
    dropout_rate,
    pretrained_weights=None,
    training=True,
):
    # build the model
    inputs = keras.layers.Input(
        shape=(img_height, img_width, img_channels)
    )

    # Contraction
    c1, p1 = conv_block_down(
        input_tensor=inputs,
        num_filters=16,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )
    c2, p2 = conv_block_down(
        input_tensor=p1,
        num_filters=32,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )
    c3, p3 = conv_block_down(
        input_tensor=p2,
        num_filters=64,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    c4, p4 = conv_block_down(
        input_tensor=p3,
        num_filters=128,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    c5, p5 = conv_block_down(
        input_tensor=p4,
        num_filters=256,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    c6, p6 = conv_block_down(
        input_tensor=p5,
        num_filters=512,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    # Expansion
    u1 = conv_block_up(
        input_tensor=c6,
        skip_tensor=c5,
        num_filters=256,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )
    u2 = conv_block_up(
        input_tensor=u1,
        skip_tensor=c4,
        num_filters=128,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    u3 = conv_block_up(
        input_tensor=u2,
        skip_tensor=c3,
        num_filters=64,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    u4 = conv_block_up(
        input_tensor=u3,
        skip_tensor=c2,
        num_filters=32,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )
    u5 = conv_block_up(
        input_tensor=u4,
        skip_tensor=c1,
        num_filters=16,
        dropout_rate=dropout_rate,
        kernel_size=(3, 3),
    )

    outputs = keras.layers.Conv2D(
        5, kernel_size=(1, 1), activation="softmax"
    )(u5)

    model = keras.Model(inputs=[inputs], outputs=[outputs], name="U-Net")

    model.summary()
    return model


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
    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer="he_normal",
        padding="same",
    )(input_tensor)

    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation("relu")(conv)

    # Second convolutional layer
    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer="he_normal",
        padding="same",
    )(conv)

    conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation("relu")(conv)

    # Dropout layer
    conv = keras.layers.Dropout(dropout_rate)(conv)

    # Max pooling layer
    pool = keras.layers.MaxPooling2D((2, 2))(conv)

    print(f"conv_block_down output shape: {conv.shape}, pool shape: {pool.shape}")

    return conv, pool

    


def conv_block_up(
    input_tensor, skip_tensor, num_filters, dropout_rate, kernel_size
):
    """
    Creates a upsampling convolutional block for U-Net architecture.

    Args:
    - input_tensor (tf.Tensor): Input tensor to the convolutional block.
    - skip_tensor (tf.Tensor): Tensor for the skip connection from the
    downsampling path.
    - num_filters (int): Number of filters for the convolutional layers.
    - dropout_rate (float): Dropout rate for regularization.
    - kernel_size (tuple): Size of the kernel for convolutional layers.

    Returns:
    - c (tf.Tensor): Output tensor from the last convolutional layer.
    """
    # First upconvolution
    u = keras.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=(2, 2),
        strides=(2, 2),
        padding="same",
        activation="relu"
    )(input_tensor)

    # Concatenating Upconvolution with Contraction tensor
    u = keras.layers.concatenate([u, skip_tensor])

    # First convolutional layer
    c = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer="he_normal",
        padding="same",
    )(u)

    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation("relu")(c)

    # Second convolutional layer
    c = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer="he_normal",
        padding="same",
    )(c)

    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation("relu")(c)

    # Dropout-Layer
    c = keras.layers.Dropout(dropout_rate)(c)

    print(f"conv_block_up output shape: {c.shape}")

    return c
