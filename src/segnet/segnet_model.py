import tensorflow as tf
from custom_layers import MaxPoolingWithIndices2D, MaxUnpooling2D
<<<<<<< HEAD
from keras.layers import Input, Conv2D, BatchNormalization
from keras.models import Model

=======
from keras.layers import BatchNormalization, Conv2D, Input
from keras.models import Model


>>>>>>> 8452abd5ada6cdf9d9638300b6475d3a06e6ceb4
def segnet(input_size):
    inputs = Input(input_size)

    # Encoder
<<<<<<< HEAD
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    p1, ind1 = MaxPoolingWithIndices2D((2, 2))(x)
    
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(p1)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    p2, ind2 = MaxPoolingWithIndices2D((2, 2))(x)
    
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(p2)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    p3, ind3 = MaxPoolingWithIndices2D((2, 2))(x)
    
    '''x = Conv2D(512, (3, 3), padding='same', activation='relu')(p3)
=======
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    p1, ind1 = MaxPoolingWithIndices2D((2, 2))(x)

    x = Conv2D(128, (3, 3), padding="same", activation="relu")(p1)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    p2, ind2 = MaxPoolingWithIndices2D((2, 2))(x)

    x = Conv2D(256, (3, 3), padding="same", activation="relu")(p2)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    p3, ind3 = MaxPoolingWithIndices2D((2, 2))(x)

    """x = Conv2D(512, (3, 3), padding='same', activation='relu')(p3)
>>>>>>> 8452abd5ada6cdf9d9638300b6475d3a06e6ceb4
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    p4, ind4 = MaxPoolingWithIndices2D((2, 2))(x)
    
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(p4)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
<<<<<<< HEAD
    p5, ind5 = MaxPoolingWithIndices2D((2, 2))(x)'''

    # Decoder
    '''x = MaxUnpooling2D((2, 2))([p5, ind5])
=======
    p5, ind5 = MaxPoolingWithIndices2D((2, 2))(x)"""

    # Decoder
    """x = MaxUnpooling2D((2, 2))([p5, ind5])
>>>>>>> 8452abd5ada6cdf9d9638300b6475d3a06e6ceb4
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxUnpooling2D((2, 2))([x, ind4])
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
<<<<<<< HEAD
    x = BatchNormalization()(x)'''
    
    x = MaxUnpooling2D((2, 2))([p3, ind3])
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxUnpooling2D((2, 2))([x, ind2])
    x = Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = MaxUnpooling2D((2, 2))([x, ind1])
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    
    outputs = Conv2D(filters=1, kernel_size=(1, 1), padding='same', activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model


=======
    x = BatchNormalization()(x)"""

    x = MaxUnpooling2D((2, 2))([p3, ind3])
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = MaxUnpooling2D((2, 2))([x, ind2])
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    x = MaxUnpooling2D((2, 2))([x, ind1])
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    outputs = Conv2D(
        filters=1, kernel_size=(1, 1), padding="same", activation="sigmoid"
    )(x)

    model = Model(inputs, outputs)

    return model
>>>>>>> 8452abd5ada6cdf9d9638300b6475d3a06e6ceb4
