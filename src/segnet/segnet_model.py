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
    print(f"Input shape: {x.shape}") 
    print(enumerate(num_filters))
    print(reversed(list(enumerate(num_filters))))

    for i, filters in enumerate(num_filters):
        print(f"\nEncoder Block {i+1}:")
        # Determine the number of convolutions for this block
        num_convs = 2 if i < 2 else (3 if i < len(num_filters) - 1 else 4)
        
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

            print(f"  After Conv {conv_idx+1}, shape: {x.shape}")

        # MaxPooling with Indices
        x, indices = MaxPoolingWithIndices(pool_size=(2, 2))(x)
        pool_indices.append((indices, filters))
        input_shapes.append(tf.shape(x))
        print(f"  After Pooling, feature map shape: {x.shape}")  # Print the feature map shape after pooling
        print(f"  Pooling indices shape: {indices.shape}")

    # Decoder
    for i, filters in reversed(list(enumerate(num_filters))):
        print(f"\nDecoder Block {i+1}:")
        indices, pool_filters = pool_indices.pop()
        # MaxUnpooling2D with indices to double the resolution
        x = MaxUnpooling2D()([x, indices])

        print(f"  After Unpooling, shape: {x.shape}") 
        # Apply the same number of convolutions as in the encoder block
        num_convs = 2 if i < 2 else (3 if i < len(num_filters) - 1 else 4)
        
        # Apply convolutional layers in the decoder block
        for conv_idx in range(num_convs):
            if initializer_function == "he_normal":
                initializer = keras.initializers.HeNormal()
            elif initializer_function == "he_uniform":
                initializer = keras.initializers.HeUniform()
            x = Conv2D(pool_filters, kernel_size, padding="same", kernel_initializer=initializer)(x)
            if use_batchnorm:
                x = BatchNormalization()(x)
            x = Activation(activation)(x) if activation != "prelu" else keras.layers.PReLU()(x)

            # Apply Dropout only between conv layers (not after the last conv in the block)
            if conv_idx < num_convs - 1:
                x = Dropout(dropout_rate)(x)
            
            print(f"  After Conv {conv_idx+1}, shape: {x.shape}") 

    outputs = Conv2D(
        5, kernel_size=(1, 1), padding="same", activation="softmax"
    )(x)

    model = Model(inputs, outputs, name="SegNet")
    model.summary()
    return model
