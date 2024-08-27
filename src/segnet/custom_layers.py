import tensorflow as tf
from keras import backend as K
from keras.layers import Layer


class MaxUnpooling2D(Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        updates, mask = inputs
        input_shape = K.shape(updates)
        output_shape = [
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        ]

        flat_input_size = tf.reduce_prod(input_shape[1:])
        flat_output_size = tf.reduce_prod(output_shape[1:])

        mask = tf.reshape(mask, [-1])
        flat_mask = mask

        batch_size = input_shape[0]
        batch_offsets = tf.range(batch_size) * flat_output_size
        batch_offsets = tf.reshape(batch_offsets, [-1, 1])
        flat_mask = flat_mask + batch_offsets

        flat_updates = tf.reshape(updates, [-1])
        flat_output = tf.zeros([batch_size * flat_output_size], dtype=updates.dtype)

        flat_output = tf.tensor_scatter_nd_add(flat_output, tf.expand_dims(flat_mask, 1), flat_updates)

        return tf.reshape(flat_output, output_shape)

    def compute_output_shape(self, input_shape):
        shape = [
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        ]
        return tf.TensorShape(shape)

    def get_config(self):
        config = super(MaxUnpooling2D, self).get_config()
        config.update({
            "pool_size": self.pool_size,
        })
        return config


class MaxPoolingWithIndices2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=(2, 2), padding="SAME", **kwargs):
        super(MaxPoolingWithIndices2D, self).__init__(**kwargs)
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding

    def call(self, inputs):
        pool, indices = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.strides[0], self.strides[1], 1],
            padding=self.padding
        )
        indices = tf.stop_gradient(indices)  # Ensure indices are not backpropagated
        return pool, indices

    def compute_output_shape(self, input_shape):
        if self.padding == "SAME":
            shape = [
                input_shape[0],
                tf.math.ceil(input_shape[1] / self.strides[0]),
                tf.math.ceil(input_shape[2] / self.strides[1]),
                input_shape[3],
            ]
        elif self.padding == "VALID":
            shape = [
                input_shape[0],
                (input_shape[1] - self.pool_size[0]) // self.strides[0] + 1,
                (input_shape[2] - self.pool_size[1]) // self.strides[1] + 1,
                input_shape[3],
            ]
        return tf.TensorShape(shape)

    def get_config(self):
        config = super(MaxPoolingWithIndices2D, self).get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config


custom_objects = {
    "MaxPoolingWithIndices2D": MaxPoolingWithIndices2D,
    "MaxUnpooling2D": MaxUnpooling2D,
}
