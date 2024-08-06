import keras
import tensorflow as tf


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
    return 1 - (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def discriminator_loss(real_output, fake_output):
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    real_loss = bce(tf.ones_like(real_output), real_output)
    fake_loss = bce(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss


def generator_loss(fake_output, gen_output, target):
    # Adversarial loss
    bce = keras.losses.BinaryCrossentropy(from_logits=False)
    adversarial_loss = bce(tf.ones_like(fake_output), fake_output)

    # Segmentation loss
    cce = keras.losses.CategoricalCrossentropy(from_logits=False)
    segmentation_loss = cce(target, gen_output)

    return adversarial_loss + segmentation_loss, segmentation_loss


def multi_scale_l1_loss(real_images, real_labels, generated_labels, intermediate_model):
    real_features = get_features(intermediate_model, [real_images, real_labels])
    generated_features = get_features(intermediate_model, [real_images, generated_labels])
    loss = keras.losses.MeanAbsoluteError()
    return tf.reduce_mean([loss(real, gen) for real, gen in zip(real_features, generated_features)])


def combined_generator_loss(critic, intermediate_model, real_images, real_labels, generated_labels):
    gen_loss, segmentation_loss = generator_loss(critic([real_images, generated_labels]), generated_labels, real_labels)
    multi_scale_loss = multi_scale_l1_loss(real_images, real_labels, generated_labels, intermediate_model)
    return gen_loss + multi_scale_loss, segmentation_loss


def combined_discriminator_loss(
    critic, intermediate_model, real_images, real_labels, generated_labels
):
    real_output = critic([real_images, real_labels])
    fake_output = critic([real_images, generated_labels])
    disc_loss = discriminator_loss(real_output, fake_output)
    multi_scale_loss = multi_scale_l1_loss(real_images, real_labels, generated_labels, intermediate_model)
    return disc_loss + multi_scale_loss


def get_features(intermediate_model, inputs):
    return intermediate_model(inputs)