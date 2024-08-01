import os
import sys

from keras import layers, models
from keras.applications import VGG16
from keras.models import Model


def discriminator(input_shape, mask_shape, filters):
    image_input = layers.Input(shape=input_shape, name="input_image")
    mask_input = layers.Input(shape=mask_shape, name="mask_image")

    x = layers.Concatenate()([image_input, mask_input])
    
    # Create layers dynamically based on the decoder filters
    for filter in filters:
        x = layers.Conv2D(filter, kernel_size=3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(
        [image_input, mask_input], outputs, name="Discriminator"
    )

    model.summary()
    return model


