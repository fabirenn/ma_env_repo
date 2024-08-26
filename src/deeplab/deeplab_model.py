import tensorflow as tf
from keras import backend as K
import keras
from keras.applications import ResNet50
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dropout,
    Input,
    UpSampling2D,
    concatenate,
)
from keras.models import Model


def atrous_spatial_pyramid_pooling(inputs, filters, dilation_rates, kernel_size):
    # Define atrous convolutions with different rates
    # Define atrous convolutions with different rates
    conv_1x1 = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation=None
    )(inputs)

    atrous_convs = [conv_1x1]
    for rate in dilation_rates:
        atrous_conv = Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            padding="same",
            activation=None,
            dilation_rate=rate,
        )(inputs)
        atrous_convs.append(atrous_conv)

    # Image-level features
    image_pooling = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
    image_pooling = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation=None
    )(image_pooling)
    image_pooling = UpSampling2D(
        size=(inputs.shape[1], inputs.shape[2]), interpolation="bilinear"
    )(image_pooling)

    # Concatenate all features
    concat = concatenate(
        [image_pooling] + atrous_convs,
        axis=3,
    )
    outputs = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation=None
    )(concat)

    return outputs


def DeepLab(input_shape, dropout_rate, filters, dilation_rates, use_batchnorm, kernel_size, initializer_function, activation):
    inputs = Input(shape=input_shape)
    if initializer_function == "he_normal":
        initializer1 = keras.initializers.HeNormal()
        initializer2 = keras.initializers.HeNormal()
        initializer3 = keras.initializers.HeNormal()
    elif initializer_function == "he_uniform":
        initializer1 = keras.initializers.HeUniform()
        initializer2 = keras.initializers.HeUniform()
        initializer3 = keras.initializers.HeNormal()

    base_model = ResNet50(
        weights="imagenet", include_top=False, input_tensor=inputs
    )

    # Extract feature maps from ResNet backbone
    x = base_model.get_layer("conv4_block6_2_relu").output

    # Atrous Spatial Pyramid Pooling
    x = atrous_spatial_pyramid_pooling(
        x, filters=filters, dilation_rates=dilation_rates, kernel_size=kernel_size
    )

    # Decoder
    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer1)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    if activation == "prelu":
        x = keras.layers.PReLU()(x)
    else:
        x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer2)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    if activation == "prelu":
        x = keras.layers.PReLU()(x)
    else:
        x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer3)(x)
    if use_batchnorm:
        x = BatchNormalization()(x)
    if activation == "prelu":
        x = keras.layers.PReLU()(x)
    else:
        x = Activation(activation)(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    output = Conv2D(5, (1, 1), padding="same", activation="softmax")(x)

    model = Model(inputs, output)
    model.summary()

    return model
