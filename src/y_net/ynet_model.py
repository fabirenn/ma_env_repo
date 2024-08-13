import os
import sys
import tensorflow as tf
from keras import layers, models
from keras.layers import Conv2D, Dropout, Input, Concatenate, BatchNormalization, MaxPooling2D, Conv2DTranspose, Cropping2D, Add
from keras.models import Model

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


def semantic_feature_extractor(input_shape, dropout_rate, name_prefix=""):
    input_tensor = layers.Input(shape=input_shape, name=name_prefix + "input")
    # Encoder Path
    c1 = Conv2D(64, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_1")(input_tensor)
    b1 = BatchNormalization(name=name_prefix + "batch_normalization_1")(c1)
    c2 = Conv2D(64, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_2")(b1)
    d1 = Dropout(dropout_rate)(c2)
    p1 = MaxPooling2D((2, 2), strides=2, name=name_prefix + "max_pooling2d_1")(d1)
    
    c3 = Conv2D(128, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_3")(p1)
    b2 = BatchNormalization(name=name_prefix + "batch_normalization_2")(c3)
    c4 = Conv2D(128, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_4")(b2)
    d2 = Dropout(dropout_rate)(c4)
    p2 = MaxPooling2D((2, 2), strides=2, name=name_prefix + "max_pooling2d_2")(d2)
    
    c5 = Conv2D(256, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_5")(p2)
    c6 = Conv2D(256, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_6")(c5)
    b3 = BatchNormalization(name=name_prefix + "batch_normalization_3")(c6)
    c7 = Conv2D(256, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_7")(b3)
    d3 = Dropout(dropout_rate)(c7)
    p3 = MaxPooling2D((2, 2), strides=2, name=name_prefix + "max_pooling2d_3")(d3)
    
    c8 = Conv2D(512, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_8")(p3)
    c9 = Conv2D(512, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_9")(c8)
    b4 = BatchNormalization(name=name_prefix + "batch_normalization_4")(c9)
    c10 = Conv2D(512, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_10")(b4)
    d4 = Dropout(dropout_rate)(c10)
    p4 = MaxPooling2D((2, 2), strides=2, name=name_prefix + "max_pooling2d_4")(d4)
    
    c11 = Conv2D(512, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_11")(p4)
    c12 = Conv2D(512, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_12")(c11)
    b5 = BatchNormalization(name=name_prefix + "batch_normalization_5")(c12)
    c13 = Conv2D(512, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_13")(b5)
    d5 = Dropout(dropout_rate)(c13)
    p5 = MaxPooling2D((2, 2), strides=2, name=name_prefix + "max_pooling2d_5")(d5)
    
    c14 = Conv2D(4096, (3, 3), padding='same', activation='relu', name=name_prefix + "conv2d_14")(p5)
    c15 = Conv2D(4096, (1, 1), padding='same', activation='relu', name=name_prefix + "conv2d_15")(c14)
    c16 = Conv2D(5, (1, 1), padding='same', activation='relu', name=name_prefix + "conv2d_16")(c15)
    
    # Decoder Path
    d1 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu', name=name_prefix + "conv2d_transpose_1")(c16)
    c17 = Conv2D(5, (1, 1), padding='same', activation='relu', name=name_prefix + "conv2d_17")(d1)
    r1 = Cropping2D(cropping=((0, 0), (0, 0)), name=name_prefix + "cropping2d_1")(c13)
    r1 = Conv2D(5, (1, 1), padding='same', activation='relu', name=name_prefix + "conv2d_18")(r1)  # Match channels to 5
    d6 = Dropout(dropout_rate)(r1)
    s1 = Add(name=name_prefix + "add_1")([d6, c17])
    
    d2 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu', name=name_prefix + "conv2d_transpose_2")(s1)
    c18 = Conv2D(5, (1, 1), padding='same', activation='relu', name=name_prefix + "conv2d_19")(d2)
    r2 = Cropping2D(cropping=((0, 0), (0, 0)), name=name_prefix + "cropping2d_2")(c10)
    r2 = Conv2D(5, (1, 1), padding='same', activation='relu', name=name_prefix + "conv2d_20")(r2)  # Match channels to 5
    d7 = Dropout(dropout_rate)(r2)
    s2 = Add(name=name_prefix + "add_2")([d7, c18])
    
    d3 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu', name=name_prefix + "conv2d_transpose_3")(s2)
    d4 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu', name=name_prefix + "conv2d_transpose_4")(d3)
    d5 = Conv2DTranspose(5, (4, 4), strides=(2, 2), padding='same', activation='relu', name=name_prefix + "conv2d_transpose_5")(d4)
    r3 = Cropping2D(cropping=((0, 0), (0, 0)), name=name_prefix + "cropping2d_3")(d5)  # Adjust the cropping values based on dimensions

    # Output
    output = Conv2D(5, (1, 1), padding='same', activation='softmax', name=name_prefix + "conv2d_21")(r3)

    return input_tensor, output


def detail_feature_extractor(input_shape, dropout_rate):
    input_tensor = layers.Input(shape=input_shape)
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c1")(input_tensor)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c2")(c1)
    c3 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c3")(c2)
    c4 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c4")(c3)
    c5 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c5")(c4)
    c6 = Conv2D(16, (3, 3), activation="relu", padding="same", name="c6")(c5)
    do1 = Dropout(dropout_rate)(c6)
    c7 = Conv2D(32, (5, 5), activation="relu", padding="same", name="c7")(do1)
    c8 = Conv2D(32, (5, 5), activation="relu", padding="same", name="c8")(c7)
    c9 = Conv2D(32, (5, 5), activation="relu", padding="same", name="c9")(c8)
    do2 = Dropout(dropout_rate)(c9)
    c10 = Conv2D(64, (5, 5), activation="relu", padding="same", name="c10")(do2)
    c11 = Conv2D(64, (5, 5), activation="relu", padding="same", name="c11")(c10)
    c12 = Conv2D(64, (5, 5), activation="relu", padding="same", name="c12")(c11)
    output = Conv2D(5, (1, 1), activation="softmax", name="c13")(c12)

    return input_tensor, output


def fusion_module(y1_output, y2_output):
    f1 = Concatenate(name="concatenate")([y1_output, y2_output])
    c1 = Conv2D(16, (3, 3), activation="relu", padding="same", name="f1")(f1)
    c2 = Conv2D(16, (3, 3), activation="relu", padding="same", name="f2")(c1)
    c3 = Conv2D(32, (3, 3), activation="relu", padding="same", name="f3")(c2)
    c4 = Conv2D(32, (3, 3), activation="relu", padding="same", name="f4")(c3)
    outputs = Conv2D(5, (1, 1), activation="softmax", name="output")(c4)
    return outputs


def build_ynet(img_width, img_height, channel_size, dropout_rate):
    input_shape = (img_width, img_height, channel_size)
    inputs = Input(shape=input_shape)
    
    # Semantic Feature Extractor
    semantic_inputs, y1_output = semantic_feature_extractor(input_shape, dropout_rate)
    pretrained_semantic_model = Model(inputs=semantic_inputs, outputs=y1_output, name='Semantic-Feature-Extractor')
    
    # Detail Feature Extractor
    detail_inputs, y2_output = detail_feature_extractor(input_shape, dropout_rate)
    detail_model = Model(inputs=detail_inputs, outputs=y2_output, name='Detail-Feature-Extractor')

    outputs = fusion_module(pretrained_semantic_model.output, detail_model.output)

    model = Model(inputs, outputs, name="Y-Net")

    model.summary()
    return model


def build_feature_extractor_for_pretraining(img_width, img_height, channel_size, dropout_rate):
    input_shape = (img_width, img_height, channel_size)
    
    # Semantic Feature Extractor
    semantic_input, y1_output = semantic_feature_extractor(input_shape, dropout_rate, name_prefix="pretrain_")

    model = Model(semantic_input, y1_output, name="Pretraining_Model")
    model.summary()
    return model

def build_ynet_with_pretrained_semantic_extractor(img_width, img_height, channel_size, dropout_rate, semantic_extractor_model):
    input_shape = (img_width, img_height, channel_size)
    inputs = Input(input_shape)
    
    # Semantic Feature Extractor
    y1_inputs, y1_output = semantic_feature_extractor(input_shape, dropout_rate)

    pretrained_y1 = Model(inputs=y1_inputs, outputs=y1_output, name='Pretrained-Semantic-Feature-Extractor')

    for layer in pretrained_y1.layers:
        if isinstance(layer, (layers.Conv2D, layers.BatchNormalization, layers.Conv2DTranspose, layers.Add, layers.Cropping2D)):
            try:
                layer.set_weights(semantic_extractor_model.get_layer("pretrain_" + layer.name).get_weights())
            
            except ValueError:
                print(f"Layer {layer.name} not found in the pretrained model. Skipping.")
    
    y2_input, y2_output = detail_feature_extractor(input_shape)

    y2 = Model(inputs=y2_input, outputs=y2_output, name="Detail-Extractor")
    
    # Fusion Module
    outputs = fusion_module(pretrained_y1(inputs), y2(inputs))

    # Model definition
    model = Model(inputs, outputs, name="Y-Net")
    model.summary()
    return model
