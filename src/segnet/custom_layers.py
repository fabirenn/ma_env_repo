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
        batch_size = input_shape[0]
        height = input_shape[1] * self.pool_size[0]
        width = input_shape[2] * self.pool_size[1]
        channels = input_shape[3]

        # Calculate output shape
        output_shape = [batch_size, height, width, channels]
        output_shape = tf.stack(output_shape)

        # Flatten the mask and updates
        flat_input_size = input_shape[1] * input_shape[2] * channels
        flat_output_size = height * width * channels

        mask = tf.cast(mask, dtype=tf.int32)
        flat_mask = tf.reshape(mask, [-1])

        # Compute the batch offsets
        batch_range = tf.reshape(tf.range(batch_size, dtype=tf.int32), shape=[-1, 1, 1, 1])
        b = tf.ones_like(mask) * batch_range
        b = tf.reshape(b, [-1])
        flat_mask += tf.cast(b * flat_input_size, dtype=tf.int32)

        # Flatten the updates
        flat_updates = tf.reshape(updates, [-1])

        # Initialize the flat output tensor
        flat_output = tf.zeros([batch_size * flat_output_size], dtype=updates.dtype)

        # Scatter the updates into the flat output tensor
        flat_output = tf.tensor_scatter_nd_add(flat_output, tf.expand_dims(flat_mask, 1), flat_updates)

        # Reshape the flat output back to the original output shape
        ret = tf.reshape(flat_output, output_shape)
        return ret

    def compute_output_shape(self, input_shape):
        shape = [
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        ]
        return tf.TensorShape(shape)


class MaxPoolingWithIndices2D(Layer):
    def __init__(
        self, pool_size=(2, 2), strides=(2, 2), padding="SAME", **kwargs
    ):
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
        
        # Flatten the indices for compatibility with the unpooling operation
        indices = tf.cast(indices, dtype=tf.int32)  # Ensure indices are int32 for compatibility
        return pool, indices

    def get_config(self):
        config = super(MaxPoolingWithIndices2D, self).get_config()
        config.update({
            "pool_size": self.pool_size,
            "strides": self.strides,
            "padding": self.padding,
        })
        return config

    def compute_output_shape(self, input_shape):
        if self.padding == "SAME":
            shape = [
                input_shape[0],
                input_shape[1] // self.strides[0],
                input_shape[2] // self.strides[1],
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


custom_objects = {
    "MaxPoolingWithIndices2D": MaxPoolingWithIndices2D,
    "MaxUnpooling2D": MaxUnpooling2D,
}
