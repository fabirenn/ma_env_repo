import tensorflow as tf
from keras import backend as K


class MaxUnpooling2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxUnpooling2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        updates, mask = inputs[0], inputs[1]
        input_shape = K.shape(updates)
        mask = tf.cast(mask, "int32")

        # Calculate output shape
        output_shape = [
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        ]

        # Calculate indices for scatter
        one_like_mask = tf.ones_like(mask, dtype="int32")
        batch_shape = tf.concat([[input_shape[0]], [1], [1], [1]], axis=0)
        batch_range = tf.reshape(
            tf.range(output_shape[0], dtype="int32"), shape=batch_shape
        )
        b = one_like_mask * batch_range

        y = mask // (output_shape[2] * output_shape[3])
        x = (mask // output_shape[3]) % output_shape[2]
        feature_range = tf.range(output_shape[3], dtype='int32')
        f = one_like_mask * feature_range

        updates_size = tf.size(updates)
        indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
        values = tf.reshape(updates, [updates_size])

        # Ensure indices are within bounds
        output_shape_tensor = tf.convert_to_tensor(output_shape, dtype=tf.int32)
        indices = tf.clip_by_value(indices, 0, output_shape_tensor - 1)

        ret = tf.scatter_nd(indices, values, output_shape)

        return ret

    def compute_output_shape(self, input_shape):
        shape = [
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        ]
        return tf.TensorShape(shape)


class MaxPoolingWithIndices2D(tf.keras.layers.Layer):
    def __init__(self, pool_size=(2, 2), **kwargs):
        super(MaxPoolingWithIndices2D, self).__init__(**kwargs)
        self.pool_size = pool_size

    def call(self, inputs):
        pool, indices = tf.nn.max_pool_with_argmax(
            inputs,
            ksize=[1, self.pool_size[0], self.pool_size[1], 1],
            strides=[1, self.pool_size[0], self.pool_size[1], 1],
            padding="SAME",
        )
        return pool, indices

    def compute_output_shape(self, input_shape):
        shape = [
            input_shape[0],
            input_shape[1] * self.pool_size[0],
            input_shape[2] * self.pool_size[1],
            input_shape[3],
        ]
        return tf.TensorShape(shape)
