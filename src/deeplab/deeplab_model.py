import tensorflow as tf
from keras import backend as K
from keras.applications import ResNet50
from keras.layers import (
    Activation,
    BatchNormalization,
    Conv2D,
    Dropout,
    Input,
    Layer,
    UpSampling2D,
    concatenate,
)
from keras.models import Model


class CRFLayer(Layer):
    def __init__(
        self,
        image_shape,
        theta_alpha=160.0,
        theta_beta=3.0,
        theta_gamma=3.0,
        num_iterations=10,
        **kwargs,
    ):
        self.image_shape = image_shape
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.num_iterations = num_iterations
        super(CRFLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(CRFLayer, self).build(input_shape)

    def call(self, inputs):
        image, logits = inputs

        # Convert logits to unary potentials
        unary = tf.nn.softmax(logits, axis=-1)
        unary = -tf.math.log(unary + 1e-10)

        # Initialize Q to unary
        Q = unary

        def spatial_kernel(x):
            return tf.image.resize(
                x, self.image_shape[:2], method=tf.image.ResizeMethod.BILINEAR
            )

        def bilateral_kernel(x):
            return tf.image.resize(
                x, self.image_shape[:2], method=tf.image.ResizeMethod.BILINEAR
            )

        # Perform mean-field inference
        for i in range(self.num_iterations):
            # Spatial term
            spatial_out = spatial_kernel(Q)
            spatial_out = K.exp(-spatial_out / (2 * self.theta_gamma**2))
            spatial_out = K.sum(spatial_out, axis=-1, keepdims=True)

            # Bilateral term
            bilateral_out = bilateral_kernel(Q)
            bilateral_out = K.exp(-bilateral_out / (2 * self.theta_alpha**2))
            bilateral_out = K.sum(bilateral_out, axis=-1, keepdims=True)

            # Update Q
            Q = unary + spatial_out + bilateral_out
            Q = tf.nn.softmax(Q, axis=-1)

        return Q

    def compute_output_shape(self, input_shape):
        return input_shape[1]


def atrous_spatial_pyramid_pooling(inputs, filters, dilation_rates):
    # Define atrous convolutions with different rates
    # Define atrous convolutions with different rates
    conv_1x1 = Conv2D(
        filters=filters, kernel_size=(1, 1), padding="same", activation=None
    )(inputs)

    atrous_convs = [conv_1x1]
    for rate in dilation_rates:
        atrous_conv = Conv2D(
            filters=filters,
            kernel_size=(3, 3),
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


def DeepLab(input_shape, dropout_rate, filters=256, dilation_rates=[1, 2, 4]):
    inputs = Input(shape=input_shape)
    base_model = ResNet50(
        weights="imagenet", include_top=False, input_tensor=inputs
    )

    # Extract feature maps from ResNet backbone
    x = base_model.get_layer("conv4_block6_2_relu").output

    # Atrous Spatial Pyramid Pooling
    x = atrous_spatial_pyramid_pooling(
        x, filters=filters, dilation_rates=dilation_rates
    )

    # Decoder
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropout_rate)(x)
    x = UpSampling2D((2, 2), interpolation="bilinear")(x)
    output = Conv2D(5, (1, 1), padding="same", activation="softmax")(x)

    model = Model(inputs, output)
    model.summary()

    crf_output = CRFLayer(input_shape)([inputs, output])
    model_crf = Model(inputs, crf_output)

    return model, model_crf
