from custom_layers import MaxPoolingWithIndices2D, MaxUnpooling2D
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input
from keras.models import Model
import keras


def segnet(input_size, dropout_rate, num_filters, kernel_size, activation, use_batchnorm, initializer_function):
    inputs = Input(input_size)
    
    # Encoder
    pool_indices = []
    input_shapes = []
    x = inputs
    for filters in num_filters:
        if initializer_function == "he_normal":
            initializer = keras.initializers.HeNormal()
        elif initializer_function == "he_uniform":
            initializer = keras.initializers.HeUniform()
        
        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)

        p = MaxPoolingWithIndices2D(pool_size=(2, 2), strides=(2, 2))(x)
        pool_indices.append(p.indices)
        input_shapes.append(p.input_shape)
        x = p

    # Decoder
    for filters, indices, input_shape in zip(reversed(num_filters), reversed(pool_indices), reversed(input_shapes)):
        # Unpooling
        x = MaxUnpooling2D(pool_size=(2, 2))(x, indices, output_shape=input_shape)

        if initializer_function == "he_normal":
            initializer = keras.initializers.HeNormal()
        elif initializer_function == "he_uniform":
            initializer = keras.initializers.HeUniform()

        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)
        x = Dropout(dropout_rate)(x)

        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)

    outputs = Conv2D(
        5, kernel_size=(1, 1), padding="same", activation="softmax"
    )(x)

    model = Model(inputs, outputs, name="SegNet")
    model.summary()
    return model
