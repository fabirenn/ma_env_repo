import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
import albumentations as A
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt



image_dir = "../data/train/original"
mask_dir = "../data/train/mask"

image_paths = sorted([os.path.join(image_dir, file) for file in os.listdir(image_dir) if file.endswith('.jpg')])
mask_paths = sorted([os.path.join(mask_dir, file) for file in os.listdir(mask_dir) if file.endswith('.png')])

X_train = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
Y_train = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in mask_paths] 
X_train_augmented = []
Y_train_augmented = []


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.GaussianBlur(p=0.2),
    A.Resize(height=256, width=256),
    # Add more transformations as needed
], additional_targets={'mask': 'image'})  # 'mask' is the key for masks in the augmented output

for i in range(len(X_train)):
    augmented = transform(image=X_train[i], mask=Y_train[i])
    X_train_augmented[i] = augmented['image']
    Y_train_augmented[i] = augmented['mask']