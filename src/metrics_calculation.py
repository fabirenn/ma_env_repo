import keras
import numpy as np
import tensorflow as tf
from data_loader import train_class_frequencies, val_class_frequencies, test_class_frequencies

num_classes = 5


def calculate_class_weights(class_counts, num_classes):
    total_pixels = sum(class_counts)
    class_weights = [total_pixels / (num_classes * count) for count in class_counts]
    return tf.constant(class_weights, dtype=tf.float32)


def pixel_accuracy(y_true, y_pred):
    #y_pred = tf.argmax(y_pred, axis=-1)
    #y_true = tf.argmax(y_true, axis=-1)
    correct_pixels = tf.reduce_sum(
        tf.cast(tf.equal(y_true, y_pred), tf.float32)
    )
    total_pixels = tf.size(y_true, out_type=tf.float32)
    return correct_pixels / total_pixels


def accuracy(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)
    y_pred = tf.argmax(y_pred, axis=-1)
    return tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred), tf.float32))


def mean_iou(y_true, y_pred, num_classes=5):
    y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=num_classes)
    y_true = tf.one_hot(tf.argmax(y_true, axis=-1), depth=num_classes)

    iou = []
    for i in range(num_classes):
        intersection = tf.reduce_sum(y_pred[..., i] * y_true[..., i])
        union = tf.reduce_sum(y_pred[..., i] + y_true[..., i]) - intersection
        iou.append(intersection / (union + keras.backend.epsilon()))
    return tf.reduce_mean(iou)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )


def precision(y_true, y_pred, num_classes=5):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    precisions = []
    for i in range(num_classes):
        true_positives = tf.reduce_sum(
            tf.cast((y_pred == i) & (y_true == i), tf.float32)
        )
        predicted_positives = tf.reduce_sum(tf.cast(y_pred == i, tf.float32))
        precisions.append(
            true_positives / (predicted_positives + keras.backend.epsilon())
        )
    return tf.reduce_mean(precisions)


def recall(y_true, y_pred, num_classes=5):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    recalls = []
    for i in range(num_classes):
        true_positives = tf.reduce_sum(
            tf.cast((y_pred == i) & (y_true == i), tf.float32)
        )
        actual_positives = tf.reduce_sum(tf.cast(y_true == i, tf.float32))
        recalls.append(
            true_positives / (actual_positives + keras.backend.epsilon())
        )
    return tf.reduce_mean(recalls)


def calculate_class_iou(y_true, y_pred, class_index):
    #y_pred = tf.argmax(y_pred, axis=-1)
    #y_true = tf.argmax(y_true, axis=-1)
    
    y_true_class = tf.cast(y_true == class_index, tf.float32)
    y_pred_class = tf.cast(y_pred == class_index, tf.float32)
    
    intersection = tf.reduce_sum(y_true_class * y_pred_class)
    union = tf.reduce_sum(y_true_class + y_pred_class) - intersection
    
    return tf.cond(union > 0, lambda: intersection / union, lambda: tf.constant(0.0))


def calculate_class_precision(y_true, y_pred, class_index):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)

    true_positives = tf.reduce_sum(tf.cast((y_pred == class_index) & (y_true == class_index), tf.float32))
    predicted_positives = tf.reduce_sum(tf.cast(y_pred == class_index, tf.float32))
    
    return true_positives / (predicted_positives + keras.backend.epsilon())


def calculate_class_recall(y_true, y_pred, class_index):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    true_positives = tf.reduce_sum(tf.cast((y_pred == class_index) & (y_true == class_index), tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(y_true == class_index, tf.float32))
    
    return true_positives / (actual_positives + keras.backend.epsilon())
