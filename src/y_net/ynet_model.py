import os
import sys

from keras import layers, models
from keras.layers import Conv2D, Input, concatenate, BatchNormalization, MaxPooling2D, Conv2DTranspose, Cropping2D
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

    # Block Y1
    # Layer C1^1
    x1 = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    
    # Layer B1^1
    x1 = BatchNormalization()(x1)
    
    # Layer C2^1
    x1 = Conv2D(64, (3, 3), padding='same', activation='relu')(x1)
    
    # Layer P1^1
    x1 = MaxPooling2D(pool_size=(2, 2), strides=2)(x1)
    
    # Layer C3^1
    x1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    
    # Layer B2^1
    x1 = BatchNormalization()(x1)
    
    # Layer C4^1
    x1 = Conv2D(128, (3, 3), padding='same', activation='relu')(x1)
    
    # Layer P2^1
    x1 = MaxPooling2D(pool_size=(2, 2), strides=2)(x1)
    
    # Layer C5^1
    x1 = Conv2D(256, (3, 3), padding='same', activation='relu')(x1)
    
    # Layer C6^1
    x1 = Conv2D(256, (3, 3), padding='same', activation='relu')(x1)
    
    # Block Y2
    # Layer B3^1
    x2 = BatchNormalization()(x1)
    
    # Layer C7^1
    x2 = Conv2D(256, (3, 3), padding='same', activation='relu')(x2)
    
    # Layer P3^1
    x2 = MaxPooling2D(pool_size=(2, 2), strides=2)(x2)
    
    # Layer C8^1
    x2 = Conv2D(512, (3, 3), padding='same', activation='relu')(x2)
    
    # Layer C9^1
    x2 = Conv2D(512, (3, 3), padding='same', activation='relu')(x2)
    
    # Layer B4^1
    x2 = BatchNormalization()(x2)
    
    # Layer C10^1
    x2 = Conv2D(512, (3, 3), padding='same', activation='relu')(x2)
    
    # Layer P4^1
    x2 = MaxPooling2D(pool_size=(2, 2), strides=2)(x2)
    
    # Layer C11^1
    x2 = Conv2D(512, (3, 3), padding='same', activation='relu')(x2)
    
    # Layer C12^1
    x2 = Conv2D(512, (3, 3), padding='same', activation='relu')(x2)
    
    # Block Y3
    # Layer B5^1
    x3 = BatchNormalization()(x2)
    
    # Layer C13^1
    x3 = Conv2D(512, (3, 3), padding='same', activation='relu')(x3)
    
    # Layer P5^1
    x3 = MaxPooling2D(pool_size=(2, 2), strides=2)(x3)
    
    # Layer C14^1
    x3 = Conv2D(4096, (3, 3), padding='same', activation='relu')(x3)
    
    # Layer C15^1
    x3 = Conv2D(4096, (1, 1), padding='same', activation='relu')(x3)
    
    # Layer C16^1
    x3 = Conv2D(5, (1, 1), padding='same', activation='relu')(x3)
    
    # Layer D1^1
    x3 = Conv2DTranspose(5, (4, 4), strides=2)(x3)
    
    # Layer C17^1
    x3 = Conv2D(5, (1, 1), padding='same', activation='relu')(x3)
    
    # Layer D3^1
    x3 = Cropping2D()(x3)
    
    
    # Layer D4^1
    x3 = Conv2D(512, (4, 4), strides=2, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    
    # Layer D5^1
    x3 = Conv2D(512, (4, 4), strides=2, padding='same')(x3)
    x3 = BatchNormalization()(x3)
    x3 = Activation('relu')(x3)
    
    # Layer C18^1
    x3 = Conv2D(2, (1, 1), padding='same', activation='softmax')(x3)
    
    # Concatenate the outputs of Y1, Y2, and Y3
    y1_score = Flatten()(x1)
    y2_score = Flatten()(x2)
    y3_score = Flatten()(x3)

    # Combine Y1, Y2, Y3 scores
    combined_scores = Concatenate()([y1_score, y2_score, y3_score])
    
    # Final output
    outputs = Dense(2, activation='softmax')(combined_scores)

    # Model definition
    model = Model(inputs, outputs, name='Y-Net')
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
    f1 = concatenate([y1_output, y2_output], name="concat")
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
    y1_output = y1(inputs)
    y2 = detail_feature_extractor(input_shape)
    y2_output = y2(inputs)

    outputs = fusion_module(y1_output, y2_output)

    model = Model(inputs, outputs, name="Y-Net")

    model.summary()
    return model
