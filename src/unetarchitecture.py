import os

import albumentations as A
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataLoader import SegmentationDataset
from unetModel import UNet

image_dir = "../data/train/original"
mask_dir = "../data/train/mask"

image_paths = sorted(
    [
        os.path.join(image_dir, file)
        for file in os.listdir(image_dir)
        if file.endswith(".jpg")
    ]
)
mask_paths = sorted(
    [
        os.path.join(mask_dir, file)
        for file in os.listdir(mask_dir)
        if file.endswith(".png")
    ]
)

X_train = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]
Y_train = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in mask_paths]
X_train_augmented = []
Y_train_augmented = []

transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.GaussianBlur(p=0.2),
        A.Resize(height=256, width=256),
        # Add more transformations as needed
    ],
    additional_targets={"mask": "image"},
)  # 'mask' is the key for masks in the augmented output

# Assuming X_train and Y_train are your images and masks respectively
for i in range(len(X_train)):
    augmented = transform(image=X_train[i], mask=Y_train[i])
    X_train_augmented.append(augmented["image"])
    Y_train_augmented.append(augmented["mask"])


segmentation_dataset = SegmentationDataset(
    image_paths=image_paths, mask_paths=mask_paths, transform=transform
)

# Create a DataLoader
dataloader = DataLoader(segmentation_dataset, batch_size=4, shuffle=True)

model = UNet(3, 1)
model.train()

# Specify the loss function and optimizer
criterion = (
    nn.BCEWithLogitsLoss()
)  # For binary segmentation, use a different loss for multi-class
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assuming `dataloader` is your DataLoader for the training dataset
num_epochs = 5  # Number of epochs to train for


# Example usage of the DataLoader in a training loop


for epoch in range(num_epochs):
    running_loss = 0.0
    for (
        images,
        masks,
    ) in dataloader:  # Iterate over batches of data from the DataLoader
        optimizer.zero_grad()  # Zero the parameter gradients

        # Forward pass: compute the model output
        outputs = model(images)

        # Compute the loss
        loss = criterion(outputs, masks)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update running loss
        running_loss += loss.item() * images.size(0)

    # Print average loss for the epoch
    epoch_loss = running_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "unet_model.pth")
