from custom_layers import MaxPoolingWithIndices2D, MaxUnpooling2D
from keras.layers import Activation, BatchNormalization, Conv2D, Dropout, Input
from keras.models import Model
import keras

INITIALIZER = ""
def segnet(input_size, dropout_rate, num_filters, kernel_size, activation, use_batchnorm, initializer_function):
    
    INITIALIZER = initializer_function
    inputs = Input(input_size)
    
    # Encoder
    pool_indices = []
    x = inputs
    for filters in num_filters:
        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=get_initializer())(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=get_initializer())(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        p, ind = MaxPoolingWithIndices2D((2, 2))(x)
        pool_indices.append(ind)
        x = p

    # Decoder
    for filters, ind in zip(reversed(num_filters), reversed(pool_indices)):
        x = MaxUnpooling2D((2, 2))([x, ind])
        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=get_initializer())(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)
        x = Dropout(dropout_rate)(x)
        x = Conv2D(filters, kernel_size, padding="same", kernel_initializer=get_initializer())(x)
        if use_batchnorm:
            x = BatchNormalization()(x)
        x = Activation(activation)(x)

    outputs = Conv2D(
        5, kernel_size=(1, 1), padding="same", activation="softmax"
    )(x)

    model = Model(inputs, outputs, name="SegNet")
    model.summary()
    return model


def get_initializer():
    if INITIALIZER == "he_normal":
        return keras.initializers.HeNormal()
    elif INITIALIZER == "he_uniform":
        return keras.initializers.HeUniform()
    else:
        return ""