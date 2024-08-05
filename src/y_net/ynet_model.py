import os
import sys
import tensorflow as tf
from keras import layers, models
from keras.layers import Conv2D, Input, Concatenate, BatchNormalization, MaxPooling2D, Conv2DTranspose, Cropping2D, Add
from keras.models import Model

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "u_net"))
)
from unet_model import unet

CHECKPOINT_PATH_UNET = "./artifacts/models/unet/unet_checkpoint.keras"


def semantic_feature_extractor(
    img_width, img_height, channel_size, dropout_rate
):
    inputs = Input(shape=(img_width, img_height, channel_size))

    # Encoder Path
    c1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    b1 = BatchNormalization()(c1)
    c2 = Conv2D(64, (3, 3), padding='same', activation='relu')(b1)
    p1 = MaxPooling2D((2, 2), strides=2)(c2)
    
    c3 = Conv2D(128, (3, 3), padding='same', activation='relu')(p1)
    b2 = BatchNormalization()(c3)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu')(b2)
    p2 = MaxPooling2D((2, 2), strides=2)(c4)
    
    c5 = Conv2D(256, (3, 3), padding='same', activation='relu')(p2)
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu')(c5)
    b3 = BatchNormalization()(c6)
    c7 = Conv2D(256, (3, 3), padding='same', activation='relu')(b3)
    p3 = MaxPooling2D((2, 2), strides=2)(c7)
    
    c8 = Conv2D(512, (3, 3), padding='same', activation='relu')(p3)
    c9 = Conv2D(512, (3, 3), padding='same', activation='relu')(c8)
    b4 = BatchNormalization()(c9)
    c10 = Conv2D(512, (3, 3), padding='same', activation='relu')(b4)
    p4 = MaxPooling2D((2, 2), strides=2)(c10)
    
    c11 = Conv2D(512, (3, 3), padding='same', activation='relu')(p4)
    c12 = Conv2D(512, (3, 3), padding='same', activation='relu')(c11)
    b5 = BatchNormalization()(c12)
    c13 = Conv2D(512, (3, 3), padding='same', activation='relu')(b5)
    p5 = MaxPooling2D((2, 2), strides=2)(c13)
    
    c14 = Conv2D(4096, (3, 3), padding='same', activation='relu')(p5)
    c15 = Conv2D(4096, (1, 1), padding='same', activation='relu')(c14)
    c16 = Conv2D(5, (1, 1), padding='same', activation='relu')(c15)
    
    # Decoder Path
    d1 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu')(c16)
    c17 = Conv2D(5, (1, 1), padding='same', activation='relu')(d1)
    r1 = Cropping2D(cropping=((0, 0), (0, 0)))(c13)
    r1 = Conv2D(5, (1, 1), padding='same', activation='relu')(r1)  # Match channels to 5
    s1 = Add()([r1, c17])
    
    d2 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu')(s1)
    c18 = Conv2D(5, (1, 1), padding='same', activation='relu')(d2)
    r2 = Cropping2D(cropping=((0, 0), (0, 0)))(c10)
    r2 = Conv2D(5, (1, 1), padding='same', activation='relu')(r2)  # Match channels to 5
    s2 = Add()([r2, c18])
    
    d3 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu')(s2)
    d4 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu')(d3)
    d5 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu')(d4)
    r3 = Cropping2D(cropping=((0, 0), (0, 0)))(d5)  # Adjust the cropping values based on dimensions

    # Output
    y1_score = Conv2D(5, (1, 1), padding='same', activation='softmax')(r3)

    # Model definition
    model = models.Model(inputs, y1_score, name='Y-Net')
    model.summary()
    return model


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
    output = Conv2D(5, (1, 1), activation="softmax", name="c13")(c12)

    model = models.Model(input, output, name="Detailed-Feature-Extractor")
    model.summary()

    return model


def fusion_module(y1_output, y2_output):
    f1 = Concatenate(name="concatenate")([y1_output, y2_output])
    # Ensure that f1 is a tensor
    f1 = tf.convert_to_tensor(f1)
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", name="f1")(f1)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same", name="f2")(c1)
    c3 = Conv2D(32, (3, 3), activation="relu", padding="same", name="f3")(c2)
    c4 = Conv2D(32, (3, 3), activation="relu", padding="same", name="f4")(c3)
    outputs = Conv2D(5, (1, 1), activation="softmax", name="output")(c4)
    return outputs


def build_ynet(img_width, img_height, channel_size, dropout_rate):
    input_shape = (img_width, img_height, channel_size)
    inputs = Input(input_shape)
    y1 = semantic_feature_extractor(
        img_width=img_width,
        img_height=img_height,
        channel_size=channel_size,
        dropout_rate=dropout_rate,
    )
    y1_output = y1.output
    y2 = detail_feature_extractor(input_shape)
    y2_output = y2.output

    outputs = fusion_module(y1_output, y2_output)

    model = Model(inputs, outputs, name="Y-Net")

    model.summary()
    return model
