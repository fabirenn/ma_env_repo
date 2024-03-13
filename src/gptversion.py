# Import necessary libraries
import os

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.nn.functional import relu
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# Define data augmentation pipeline
def get_augmentation(phase):
    list_transforms = []
    if phase == "train":
        list_transforms.extend(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.GaussianBlur(p=0.3),
                A.Resize(480, 720),
            ]
        )
    list_transforms.extend([A.Normalize(), ToTensorV2()])
    return A.Compose(list_transforms)


# Dataset class
class Dataset(Dataset):
    def __init__(self, images_dir, masks_dir, phase):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.images = os.listdir(images_dir)
        self.augmentations = get_augmentation(phase)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(
            self.masks_dir, self.images[idx].replace(".png", "_mask.png")
        )  # Assuming mask has _mask suffix
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[
            mask == 255.0
        ] = 1.0  # Assuming mask has two values: 0 (background) and 255 (foreground)

        augmented = self.augmentations(image=image, mask=mask)
        image = augmented["image"]
        mask = augmented["mask"]
        return image, mask


class UNet(nn.Module):
    def create_block(
        in_channel: int,
        mid_channel: int,
        out_channel: int,
    ) -> nn.Module:
        conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=3, padding=1)
        conv2 = nn.Conv2d(mid_channel, out_channel, kernel_size=3, padding=1)
        pool = nn.MaxPool2d(kernel_size=2, stride=2)

        return nn.Sequential(conv1, conv2, pool)

    def __init__(self, n_class):
        super().__init__()

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 572x572x3
        self.e11 = nn.Conv2d(
            3, 64, kernel_size=3, padding=1
        )  # output: 570x570x64
        self.e12 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1
        )  # output: 568x568x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 284x284x64

        # input: 284x284x64
        self.e21 = nn.Conv2d(
            64, 128, kernel_size=3, padding=1
        )  # output: 282x282x128
        self.e22 = nn.Conv2d(
            128, 128, kernel_size=3, padding=1
        )  # output: 280x280x128
        self.pool2 = nn.MaxPool2d(
            kernel_size=2, stride=2
        )  # output: 140x140x128

        # input: 140x140x128
        self.e31 = nn.Conv2d(
            128, 256, kernel_size=3, padding=1
        )  # output: 138x138x256
        self.e32 = nn.Conv2d(
            256, 256, kernel_size=3, padding=1
        )  # output: 136x136x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 68x68x256

        # input: 68x68x256
        self.e41 = nn.Conv2d(
            256, 512, kernel_size=3, padding=1
        )  # output: 66x66x512
        self.e42 = nn.Conv2d(
            512, 512, kernel_size=3, padding=1
        )  # output: 64x64x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 32x32x512

        # input: 32x32x512
        self.e51 = nn.Conv2d(
            512, 1024, kernel_size=3, padding=1
        )  # output: 30x30x1024
        self.e52 = nn.Conv2d(
            1024, 1024, kernel_size=3, padding=1
        )  # output: 28x28x1024

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(64, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out


# Training function (just as a placeholder for now)
def train_model(model, loader, optimizer, criterion, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for images, masks in tqdm(loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()


# Add validation handling to the training function
def train_and_validate_model(
    model, train_loader, val_loader, optimizer, criterion, num_epochs
):
    best_val_loss = float("inf")
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for images, masks in tqdm(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in tqdm(val_loader):
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        val_loss = val_loss / len(val_loader.dataset)

        # Print training and validation loss
        print(
            f"Epoch {epoch+1}, Train Loss: {train_loss}, Validation Loss: {val_loss}"
        )

        # Save model if validation loss has decreased
        if val_loss < best_val_loss:
            print(
                "Validation loss decreased ({:.6f} --> {:.6f}). Saving model ...".format(
                    best_val_loss, val_loss
                )
            )
            torch.save(model.state_dict(), "unet_model.pth")
            best_val_loss = val_loss


train_images_dir = "../data/train/original"
train_masks_dir = "../data/train/mask"
train_dataset = Dataset(
    images_dir=train_images_dir, masks_dir=train_masks_dir, phase="train"
)
batch_size = 1  # Set this based on your hardware constraints
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=1
)
model = UNet(1)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
num_epochs = 10

train_model(model, train_loader, optimizer, criterion, num_epochs)
