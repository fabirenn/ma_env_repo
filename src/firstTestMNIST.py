# Importieren der notwendigen Bibliotheken
import numpy as np
import tensorflow as tf
from keras import layers, models
import wandb
from wandb.keras import WandbCallback
from keras.datasets import mnist
from keras.utils import to_categorical


# W&B initialisieren
wandb.init(project='mnist_cnn', entity='fabio-renn')

# Konfigurationen für W&B
config = wandb.config
config.learning_rate = 0.01
config.epochs = 6
config.batch_size = 128

# MNIST-Daten laden und vorbereiten
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# CNN-Modell erstellen
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Modell kompilieren
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Modell trainieren
model.fit(train_images, train_labels, epochs=config.epochs, batch_size=config.batch_size,
          validation_data=(test_images, test_labels), callbacks=[WandbCallback()])

# Modellbewertung
test_loss, test_acc = model.evaluate(test_images, test_labels)
wandb.log({'test_loss': test_loss, 'test_accuracy': test_acc})

# Modell speichern (optional)
model.save('mnist_cnn_model.h5')
wandb.save('mnist_cnn_model.h5')
