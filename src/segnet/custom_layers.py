import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs, indices, output_shape):
        indices = tf.cast(indices, dtype=tf.int32)
        output_shape = tf.cast(output_shape, dtype=tf.int32)
        flat_inputs = tf.reshape(inputs, [-1])
        flat_indices = tf.reshape(indices, [-1])
        total_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
        batch_offset = tf.range(output_shape[0]) * output_shape[1] * output_shape[2] * output_shape[3]
        batch_offset = tf.reshape(batch_offset, (-1, 1))
        batch_offset = tf.reshape(tf.tile(batch_offset, [1, output_shape[1] * output_shape[2] * output_shape[3]]), [-1])
        flat_indices = flat_indices + batch_offset
        output = tf.scatter_nd(tf.expand_dims(flat_indices, axis=-1), flat_inputs, [total_elements])
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
