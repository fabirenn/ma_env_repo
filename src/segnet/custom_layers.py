import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, indices, output_shape):

        # Ensure that indices and output_shape are the correct dtype
        indices = tf.cast(indices, dtype=tf.int32)
        output_shape = tf.cast(output_shape, dtype=tf.int32)

        # Flatten the input and indices
        input_shape = tf.shape(inputs)

        # Calculate the batch size and dimension sizes
        batch_size = input_shape[0]
        height = output_shape[1]
        width = output_shape[2]
        channels = output_shape[3]

        # Create the base indices for each batch
        batch_range = tf.range(batch_size, dtype=indices.dtype)
        batch_range = tf.reshape(batch_range, [batch_size, 1, 1, 1])
        b = tf.ones_like(indices) * batch_range
        b = tf.reshape(b, [-1])

        flat_indices = tf.reshape(indices, [-1])
        flat_indices = flat_indices + (b * height * width * channels)

        flat_inputs = tf.reshape(inputs, [-1])
        ret = tf.scatter_nd(tf.expand_dims(flat_indices, axis=-1), flat_inputs, [batch_size * height * width * channels])
        ret = tf.reshape(ret, [batch_size, height, width, channels])
        
        return ret

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        )


class MaxPoolingWithIndices(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='VALID', **kwargs):
        super(MaxPoolingWithIndices, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        # Use max pooling with indices
        pooled, indices = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1, *self.pool_size, 1], strides=[1, *self.strides, 1], padding=self.padding
        )
        # Store the indices and input shape in a dictionary
        indices = tf.cast(indices, dtype=tf.int32)
        return pooled, indices

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


custom_objects = {
    "MaxPoolingWithIndices2D": MaxPoolingWithIndices,
    "MaxUnpooling2D": MaxUnpooling2D,
}
