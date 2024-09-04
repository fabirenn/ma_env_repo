from custom_layers import MaxPoolingWithIndices, MaxUnpooling2D
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input
from keras.models import Model
import keras
import tensorflow as tf


def segnet(input_size, dropout_rate, num_filters, kernel_size, activation, use_batchnorm, initializer_function):
    inputs = Input(input_size)
    pool_indices = []
    input_shapes = []
    x = inputs

    for i, filters in enumerate(num_filters):
        # Determine the number of convolutions for this block
        if i < 2:
            num_convs = 2  # First two blocks have two convolutions each
        elif i < len(num_filters) - 1:
            num_convs = 3  # The next blocks have three convolutions
        else:
            num_convs = 4  # The final block has four convolutions
        
        # Apply convolutional layers
        for conv_idx in range(num_convs):
            if initializer_function == "he_normal":
                initializer = keras.initializers.HeNormal()
            elif initializer_function == "he_uniform":
                initializer = keras.initializers.HeUniform()
            x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
            if use_batchnorm:
                x = BatchNormalization()(x)
            x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)

            # Apply Dropout only between conv layers (not after the last conv in the block)
            if conv_idx < num_convs - 1:
                x = Dropout(dropout_rate)(x)

        # MaxPooling with Indices
        x, indices = MaxPoolingWithIndices(pool_size=(2, 2), strides=(2, 2))(x)
        pool_indices.append(indices)
        input_shapes.append(tf.shape(x))

    # Decoder
    for i, filters in reversed(list(enumerate(num_filters))):
        # MaxUnpooling2D with indices to double the resolution
        x = MaxUnpooling2D(pool_size=(2, 2))(x, pool_indices[i], output_shape=input_shapes[i])

        # Apply the same number of convolutions as in the encoder block
        if i < 2:
            num_convs = 2  # First two blocks had two convolutions
        elif i < len(num_filters) - 1:
            num_convs = 3  # The next blocks had three convolutions
        else:
            num_convs = 4  # The last block had four convolutions
        
        # Apply convolutional layers in the decoder block
        for conv_idx in range(num_convs):
            if initializer_function == "he_normal":
                initializer = keras.initializers.HeNormal()
            elif initializer_function == "he_uniform":
                initializer = keras.initializers.HeUniform()
            x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
            if use_batchnorm:
                x = BatchNormalization()(x)
            x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)

            # Apply Dropout only between conv layers (not after the last conv in the block)
            if conv_idx < num_convs - 1:
                x = Dropout(dropout_rate)(x)

    outputs = Conv2D(
        5, kernel_size=(1, 1), padding="same", activation="softmax"
    )(x)

    model = Model(inputs, outputs, name="SegNet")
    model.summary()
    return model
