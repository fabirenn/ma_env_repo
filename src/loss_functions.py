import tensorflow as tf
import keras


def iou_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    iou = (intersection + 1e-10) / (union + 1e-10)

    return 1 - iou


def combined_loss(y_true, y_pred):
    bce = keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def discriminator_loss(real_output, fake_output):
    real_loss = keras.losses.BinaryCrossentropy(from_logits=False)(
        tf.ones_like(real_output), real_output
    )
    fake_loss = keras.losses.BinaryCrossentropy(from_logits=False)(
        tf.zeros_like(fake_output), fake_output
    )
    return real_loss + fake_loss


def generator_loss(fake_output):
    cce = keras.losses.CategoricalCrossentropy(from_logits=False)
    return cce(tf.ones_like(fake_output), fake_output)