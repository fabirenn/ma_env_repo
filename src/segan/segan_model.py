import os
import sys

import tensorflow as tf
from keras import layers, models
from keras.layers import BatchNormalization, Conv2D, Input
from keras.models import Model

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "segnet"))
)
from custom_layers import MaxPoolingWithIndices2D, MaxUnpooling2D
from segnet_model import segnet

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "u_net"))
)
from unet_architecture_hcp import unet


def generator(img_width, img_height, img_channels, batch_size, unet):
    if unet:
        model = unet(img_width, img_height, img_channels, batch_size)
        unet = True
    else:
        model = segnet((img_width, img_height, img_channels))
        unet = False
    return model


def discriminator(input_shape, mask_shape):
    image_input = layers.Input(shape=input_shape, name="input_image")
    mask_input = layers.Input(shape=mask_shape, name="mask_image")

    x = layers.Concatenate()([image_input, mask_input])
    x = layers.Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Conv2D(512, kernel_size=3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(
        [image_input, mask_input], outputs, name="Discriminator"
    )

    model.summary()
    return model
