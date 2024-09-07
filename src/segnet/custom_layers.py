import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, indices, output_shape):
        # Cast indices and output_shape to int32 for compatibility
        indices = tf.cast(indices, dtype=tf.int32)
        output_shape = tf.cast(output_shape, dtype=tf.int32)

        # Flatten inputs and indices
        flat_inputs = tf.reshape(inputs, [-1])
        flat_indices = tf.reshape(indices, [-1])

        # Compute batch size, height, width, and channels for the output
        batch_size, height, width, channels = output_shape[0], output_shape[1], output_shape[2], output_shape[3]

        # Calculate the total number of elements in the output space
        total_elements = height * width * channels * batch_size

        # Compute the batch offset to correctly apply indices in a batched manner
        batch_offset = tf.range(batch_size) * (height * width * channels)
        batch_offset = tf.reshape(batch_offset, (-1, 1))
        batch_offset = tf.tile(batch_offset, [1, tf.shape(flat_indices)[0] // batch_size])
        batch_offset = tf.reshape(batch_offset, [-1])

        # Add batch offset to the flat indices
        flat_indices = flat_indices + batch_offset

        # Initialize the output tensor with zeros
        output = tf.zeros([total_elements], dtype=flat_inputs.dtype)

        # Scatter the flat inputs into the output tensor based on the indices
        output = tf.tensor_scatter_nd_add(output, tf.expand_dims(flat_indices, axis=-1), flat_inputs)

        # Reshape the output tensor to the desired shape (batch_size, height, width, channels)
        output = tf.reshape(output, output_shape)

        return output


class MaxPoolingWithIndices(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding='VALID', **kwargs):
        super(MaxPoolingWithIndices, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        pooled, indices = tf.nn.max_pool_with_argmax(
            inputs, ksize=[1, *self.pool_size, 1], strides=[1, *self.strides, 1], padding=self.padding
        )
        indices = tf.cast(indices, dtype=tf.int32)
        return pooled, indices


custom_objects = {
    "MaxPoolingWithIndices2D": MaxPoolingWithIndices,
    "MaxUnpooling2D": MaxUnpooling2D,
}
