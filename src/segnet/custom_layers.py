import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, indices, output_shape):
        # Ensure indices and output_shape are the correct dtype
        indices = tf.cast(indices, dtype=tf.int32)
        output_shape = tf.cast(output_shape, dtype=tf.int32)

        # Calculate the batch size, height, width, and number of channels
        batch_size = output_shape[0]
        height = output_shape[1]
        width = output_shape[2]
        channels = output_shape[3]

        # Calculate the total number of elements
        total_elements = batch_size * height * width * channels

        # Flatten the inputs and indices
        flat_inputs = tf.reshape(inputs, [-1])
        flat_indices = tf.reshape(indices, [-1])

        # Adjust flat_indices to account for the batch offset
        batch_offset = tf.range(batch_size) * height * width * channels
        batch_offset = tf.reshape(batch_offset, (-1, 1))
        batch_offset = tf.reshape(tf.tile(batch_offset, [1, height * width * channels]), [-1])

        flat_indices = flat_indices + batch_offset

        # Scatter the flattened inputs back to the original unpooled size
        output = tf.scatter_nd(tf.expand_dims(flat_indices, axis=-1), flat_inputs, [total_elements])

        # Reshape the output to the desired output shape
        output = tf.reshape(output, output_shape)

        return output

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
