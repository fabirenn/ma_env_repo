import tensorflow as tf
from custom_layers import MaxPoolingWithIndices2D, MaxUnpooling2D
from keras.layers import Input, Conv2D, BatchNormalization
from keras.models import Model

def segnet(input_size):
    inputs = Input(input_size)

    # Encoder
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
    p5, ind5 = MaxPoolingWithIndices2D((2, 2))(x)'''

    # Decoder
    '''x = MaxUnpooling2D((2, 2))([p5, ind5])
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


