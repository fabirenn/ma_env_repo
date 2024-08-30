import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, indices, output_shape):
        # Flatten indices and inputs
        input_shape = tf.shape(inputs)
        flat_input_size = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        # Create a flat version of indices
        flat_indices = K.flatten(indices)

        # Flattened input and indices to create sparse tensor
        flat_inputs = K.flatten(inputs)
        batch_range = tf.reshape(tf.range(output_shape[0], dtype=flat_indices.dtype), shape=[-1, 1, 1, 1])
        batch_indices = batch_range * flat_output_shape[1]

        flat_indices = flat_indices + batch_indices
        ret = tf.scatter_nd(tf.expand_dims(flat_indices, axis=-1), flat_inputs, flat_output_shape)
        
        # Reshape back to original output shape
        ret = tf.reshape(ret, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] * self.pool_size[0], input_shape[2] * self.pool_size[1], input_shape[3])

    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
        })
        return config


class MaxPoolingWithIndices2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='VALID', **kwargs):
        super(MaxPoolingWithIndices2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        # Use max pooling with indices
        pooled, indices = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1, *self.pool_size, 1], strides=[1, *self.strides, 1], padding=self.padding
        )
        # Store the indices and input shape in a dictionary
        self.pooling_info = {
            "indices": indices,
            "input_shape": tf.shape(inputs)
        }
        return pooled

    def compute_output_shape(self, input_shape):
        if self.padding == 'SAME':
            output_shape = [
                input_shape[0],
                input_shape[1] // self.strides[0],
                input_shape[2] // self.strides[1],
                input_shape[3],
            ]
        else:
            output_shape = [
                input_shape[0],
                (input_shape[1] - self.pool_size[0]) // self.strides[0] + 1,
                (input_shape[2] - self.pool_size[1]) // self.strides[1] + 1,
                input_shape[3],
            ]
        return tuple(output_shape)

    def get_config(self):
        config = super(MaxPoolingWithIndices2D, self).get_config()
        config.update({
            'pool_size': self.pool_size,
            'strides': self.strides,
            'padding': self.padding
        })
        return config


custom_objects = {
    "MaxPoolingWithIndices2D": MaxPoolingWithIndices2D,
    "MaxUnpooling2D": MaxUnpooling2D,
}
