import os
import sys

from keras import layers, models
from keras.layers import Conv2D, Input, concatenate
from keras.models import Model

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "u_net"))
)
from unet_architecture_hcp import unet

CHECKPOINT_PATH_UNET = "./artifacts/models/unet/unet_checkpoint.h5"


def semantic_feature_extractor(
    img_width, img_height, batch_size, channel_size, dropout_rate
):
    unet_model = unet(
        img_width=img_width,
        img_height=img_height,
        img_channels=channel_size,
        batch_size=batch_size,
        dropout_rate=dropout_rate,
        training=True,
    )
    unet_model.load_weights(CHECKPOINT_PATH_UNET)
    return unet_model


def detail_feature_extractor(input_shape):
    input = layers.Input(shape=input_shape, name="input_shape")
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c1")(input)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c2")(c1)
    c3 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c3")(c2)
    c4 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c4")(c3)
    c5 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c5")(c4)
    c6 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c6")(c5)
    c7 = Conv2D(32, (5, 5), activation="relu", padding="same", name="c7")(c6)
    c8 = Conv2D(32, (5, 5), activation="relu", padding="same", name="c8")(c7)
    c9 = Conv2D(32, (5, 5), activation="relu", padding="same", name="c9")(c8)
    c10 = Conv2D(64, (5, 5), activation="relu", padding="same", name="c10")(c9)
    c11 = Conv2D(64, (5, 5), activation="relu", padding="same", name="c11")(c10)
    c12 = Conv2D(64, (5, 5), activation="relu", padding="same", name="c12")(c11)
    output = Conv2D(1, (1, 1), activation="sigmoid", name="c13")(c12)

    model = models.Model(input, output, name="Detailed-Feature-Extractor")
    model.summary()

    return model


def fusion_module(y1_output, y2_output):
    f1 = concatenate([y1_output, y2_output], name="concat")
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", name="f1")(f1)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same", name="f2")(c1)
    c3 = Conv2D(32, (3, 3), activation="relu", padding="same", name="f3")(c2)
    c4 = Conv2D(32, (3, 3), activation="relu", padding="same", name="f4")(c3)
    outputs = Conv2D(1, (1, 1), activation="sigmoid", name="output")(c4)
    return outputs


def build_ynet(img_width, img_height, batch_size, channel_size, dropout_rate):
    input_shape = (img_width, img_height, channel_size)
    inputs = Input(input_shape)
    y1 = semantic_feature_extractor(
        img_width=img_width,
        img_height=img_height,
        batch_size=batch_size,
        channel_size=channel_size,
        dropout_rate=dropout_rate,
    )
    y1_output = y1(inputs)
    y2 = detail_feature_extractor(input_shape)
    y2_output = y2(inputs)

    outputs = fusion_module(y1_output, y2_output)

    model = Model(inputs, outputs, name="Y-Net")

    model.summary()
    return model
