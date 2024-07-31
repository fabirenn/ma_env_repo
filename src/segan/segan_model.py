import os
import sys

from keras import layers, models
from keras.applications import VGG16
from keras.models import Model


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


def vgg_model():
    vgg = VGG16(weights="imagenet", include_top=False)
    # Definiert die Layer, aus denen die Merkmale extrahiert werden sollen
    selected_layers = [vgg.layers[i].output for i in [3, 6, 10]]
    model = Model(inputs=vgg.input, outputs=selected_layers)
    # Stelle sicher, dass die Gewichte des vortrainierten Modells nicht aktualisiert werden
    model.trainable = False
    return model
