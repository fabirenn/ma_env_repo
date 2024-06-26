import tensorflow as tf


def unet(
    img_width, img_height, img_channels, batch_size, pretrained_weights=None
):
    # build the model
    inputs = tf.keras.layers.Input(
        shape=(img_height, img_width, img_channels), batch_size=batch_size
    )

    # Contraction
    c1, p1 = conv_block_down(
        input_tensor=inputs,
        num_filters=16,
        dropout_rate=0.1,
        kernel_size=(3, 3),
    )
    c2, p2 = conv_block_down(
        input_tensor=p1, num_filters=32, dropout_rate=0.1, kernel_size=(3, 3)
    )
    c3, p3 = conv_block_down(
        input_tensor=p2, num_filters=64, dropout_rate=0.1, kernel_size=(3, 3)
    )

    # Expansion
    u1 = conv_block_up(
        input_tensor=c3,
        skip_tensor=c2,
        num_filters=32,
        dropout_rate=0.1,
        kernel_size=(3, 3),
    )
    u2 = conv_block_up(
        input_tensor=u1,
        skip_tensor=c1,
        num_filters=16,
        dropout_rate=0.1,
        kernel_size=(3, 3),
    )

    outputs = tf.keras.layers.Conv2D(
        filters=1, kernel_size=(1, 1), activation="sigmoid"
    )(u2)

    # config = wandb.config
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics="accuracy"
    )
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
    - skip_tensor (tf.Tensor): Tensor for the skip connection from the
    downsampling path.
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
