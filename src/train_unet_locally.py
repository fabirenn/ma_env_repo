import tensorflow as tf
from keras.callbacks import ModelCheckpoint, CSVLogger
from custom_callbacks import ValidationCallback
from data_loader import (
    convert_to_tensor,
    create_dataset,
    load_images_from_directory,
    load_masks_from_directory,
    make_binary_masks,
    normalize_image_data,
    resize_images,
)
from unet_model_local import unet

TRAIN_IMG_PATH = "data/generated/"
TRAIN_MASK_PATH = "data/segmented/mask/"
CHECKPOINT_PATH = "artifacts/models/test/"

IMG_WIDTH = 1024
IMG_HEIGHT = 704
IMG_CHANNEL = 3

BATCH_SIZE = 8
EPOCHS = 10

# loading images and masks from their corresponding paths into to separate lists
train_images = load_images_from_directory(TRAIN_IMG_PATH)
train_masks = load_masks_from_directory(TRAIN_MASK_PATH)


# resizing the images to dest size for training
train_images = resize_images(train_images, IMG_WIDTH, IMG_HEIGHT)
train_masks = resize_images(train_masks, IMG_WIDTH, IMG_HEIGHT)

# normalizing the values of the images and binarizing the image masks
train_images = normalize_image_data(train_images)
train_masks = make_binary_masks(train_masks, 30)

# converting the images/masks to tensors + expanding the masks tensor slide to 1 dimension
tensor_train_images = convert_to_tensor(train_images)
tensor_train_masks = convert_to_tensor(train_masks)
tensor_train_masks = tf.expand_dims(tensor_train_masks, axis=-1)

# create dataset for training purposes
train_dataset = create_dataset(
    tensor_train_images,
    tensor_train_masks,
    batchsize=BATCH_SIZE,
    buffersize=tf.data.AUTOTUNE,
)


# create model & start training it
model = unet(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNEL, BATCH_SIZE)


model.fit(
    train_dataset,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=train_dataset,
    callbacks=[
        CSVLogger(filename="logs/training_metrics.log"),
        ModelCheckpoint(
            filepath=CHECKPOINT_PATH,
            save_best_only=True,
            save_weights_only=True,
            monitor="val_accuracy"
        ),
        ValidationCallback(model=model, validation_data=train_dataset),
    ],
)



