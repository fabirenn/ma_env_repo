import keras
import tensorflow as tf
from keras.layers import BatchNormalization


def unet(
    img_width,
    img_height,
    img_channels,
    dropout_rate,
    filters_list,
    kernel_size, 
    activation, 
    use_batchnorm, 
    initializer_function,
    pretrained_weights=None,
    training=True,
):
    # build the model
    inputs = keras.layers.Input(shape=(img_height, img_width, img_channels))

    # Contraction
    c, p = [], []
    for i, filters in enumerate(filters_list):
        if i == 0:
            input_tensor = inputs
        else:
            input_tensor = p[i - 1]
        c_layer, p_layer = conv_block_down(
            input_tensor=input_tensor,
            num_filters=filters,
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            activation=activation, 
            use_batchnorm=use_batchnorm,
            initializer_function=initializer_function
        )
        c.append(c_layer)
        p.append(p_layer)

    # Expansion
    u = c[-1]
    for i in range(len(filters_list) - 2, -1, -1):
        u = conv_block_up(
            input_tensor=u,
            skip_tensor=c[i],
            num_filters=filters_list[i],
            dropout_rate=dropout_rate,
            kernel_size=kernel_size,
            activation=activation, 
            use_batchnorm=use_batchnorm,
            initializer_function=initializer_function
        )

    outputs = keras.layers.Conv2D(5, kernel_size=(1, 1), activation="softmax")(
        u
    )

    model = keras.Model(inputs=[inputs], outputs=[outputs], name="U-Net")

    model.summary()
    return model


def conv_block_down(input_tensor, num_filters, dropout_rate, kernel_size, activation, use_batchnorm, initializer_function):
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
        kernel_initializer=initializer_function,
        padding="same",
    )(input_tensor)
    if use_batchnorm:
        conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation(activation)(conv)

    # Second convolutional layer
    conv = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer=initializer_function,
        padding="same",
    )(conv)

    if use_batchnorm:
        conv = keras.layers.BatchNormalization()(conv)
    conv = keras.layers.Activation(activation)(conv)

    # Dropout layer
    conv = keras.layers.Dropout(dropout_rate)(conv)

    # Max pooling layer
    pool = keras.layers.MaxPooling2D((2, 2))(conv)

    # print(f"conv_block_down output shape: {conv.shape}, pool shape: {pool.shape}")

    return conv, pool


def conv_block_up(
    input_tensor, skip_tensor, num_filters, dropout_rate, kernel_size, activation, use_batchnorm, initializer_function
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
        activation=activation,
    )(input_tensor)

    # Concatenating Upconvolution with Contraction tensor
    u = keras.layers.concatenate([u, skip_tensor])

    # First convolutional layer
    c = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer=initializer_function,
        padding="same",
    )(u)
    if use_batchnorm:
        c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation(activation)(c)

    # Second convolutional layer
    c = keras.layers.Conv2D(
        num_filters,
        kernel_size,
        kernel_initializer=initializer_function,
        padding="same",
    )(c)

    if use_batchnorm:
        c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation(activation)(c)

    # Dropout-Layer
    c = keras.layers.Dropout(dropout_rate)(c)

    # print(f"conv_block_up output shape: {c.shape}")

    return c
