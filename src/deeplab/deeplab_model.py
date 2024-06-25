import numpy as np
import pydensecrf.densecrf as dcrf
import tensorflow as tf
from keras.applications import ResNet50
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
from pydensecrf.utils import (
    create_pairwise_bilateral,
    create_pairwise_gaussian,
    unary_from_softmax,
)


def atrous_spatial_pyramid_pooling(inputs, filters):
    # Define atrous convolutions with different rates
    conv_1x1 = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation="relu"
    )(inputs)
    conv_3x3_rate_6 = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        dilation_rate=6,
    )(inputs)
    conv_3x3_rate_12 = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        dilation_rate=12,
    )(inputs)
    conv_3x3_rate_18 = Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        padding="same",
        activation="relu",
        dilation_rate=18,
    )(inputs)

    # Image-level features
    image_pooling = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
    image_pooling = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation="relu"
    )(image_pooling)
    image_pooling = UpSampling2D(
        size=(inputs.shape[1], inputs.shape[2]), interpolation="bilinear"
    )(image_pooling)

    # Concatenate all features
    concat = concatenate(
        [
            conv_1x1,
            conv_3x3_rate_6,
            conv_3x3_rate_12,
            conv_3x3_rate_18,
            image_pooling,
        ],
        axis=-1,
    )
    outputs = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation="relu"
    )(concat)

    return outputs


def apply_crf(image, prediction):
    """
    Apply CRF to the prediction.

    Parameters:
    image: The original image
    prediction: The model's softmax output

    Returns:
    result: The CRF-refined segmentation
    """
    # Convert softmax output to unary potentials
    unary = unary_from_softmax(prediction)
    unary = np.ascontiguousarray(unary)

    # Create the dense CRF model
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], prediction.shape[0])
    d.setUnaryEnergy(unary)

    # Create pairwise potentials (bilateral and spatial)
    pairwise_gaussian = create_pairwise_gaussian(
        sdims=(3, 3), shape=image.shape[:2]
    )
    d.addPairwiseEnergy(pairwise_gaussian, compat=3)

    pairwise_bilateral = create_pairwise_bilateral(
        sdims=(50, 50), schan=(20, 20, 20), img=image, chdim=2
    )
    d.addPairwiseEnergy(pairwise_bilateral, compat=10)

    # Perform inference
    Q = d.inference(5)

    # Convert the Q array to the final prediction
    result = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))
    return result


def DeepLab(input_shape):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_tensor=inputs
    )

    # Extract feature maps from ResNet backbone
    x = base_model.get_layer("conv4_block6_2_relu").output

    # Atrous Spatial Pyramid Pooling
    x = atrous_spatial_pyramid_pooling(x, filters=256)

    # Decoder
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(x)

    model = Model(inputs, outputs)
    model.summary()

    return model
