"""
This module trains a ResNet50 model on a flower dataset with 102 classes.
It includes data loading, model setup, and training functions.
"""

import os
import scipy.io
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_labels(mat_file):
    """
    Load labels from a .mat file.

    Args:
        mat_file (str): Path to the .mat file containing labels.

    Returns:
        np.ndarray: Array of labels, adjusted to zero-based indexing if needed.
    """
    data = scipy.io.loadmat(mat_file)
    return data["labels"].squeeze() - 1


class FlowerDataset(Dataset):
    """
    Custom dataset class for flower images and labels.
    """

    def __init__(self, img_dir, labels, transform=None):
        """
        Initialize the dataset.

        Args:
            img_dir (str): Directory containing images.
            labels (np.ndarray): Array of labels for each image.
            transform (callable, optional): Transform to be applied on an image.
        """
        self.img_dir = img_dir
        self.labels = labels
        self.transform = transform
        self.image_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".jpg")]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def train_one_epoch(model, train_loader, criterion, optimizer, scaler=None):
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): Model to train.
        train_loader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scaler (torch.cuda.amp.GradScaler, optional): Scaler for mixed-precision training.

    Returns:
        float: Average loss over the epoch.
    """
    model.train()
    running_loss = 0.0

    for images, labels in tqdm(train_loader, desc="Training", unit="batch"):
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass with or without mixed precision
        if scaler:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(train_loader.dataset)


def create_dataloader(img_dir, labels, transform):
    """
    Create a DataLoader for the flower dataset.

    Args:
        img_dir (str): Path to the image directory.
        labels (np.ndarray): Array of image labels.
        transform (callable): Transformation to apply to each image.

    Returns:
        DataLoader: DataLoader instance for the dataset.
    """
    dataset = FlowerDataset(img_dir, labels, transform=transform)
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4 if torch.cuda.is_available() else 0,
        pin_memory=True,
    )


def main():
    """
    Main function for setting up the dataset, model, and training process.
    """
    img_dir = "data/flowers-102/jpg"
    label_file = "data/flowers-102/imagelabels.mat"
    labels = load_labels(label_file)

    # Define data transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Create DataLoader
    train_loader = create_dataloader(img_dir, labels, transform)

    # Load pre-trained ResNet model and modify the classifier
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    num_classes = len(np.unique(labels))

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-4)

    # Initialize mixed-precision scaler if using CUDA
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    # Training loop
    epochs = 1
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # Save the trained model parameters
    save_path = "flower_classification_resnet.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model parameters saved to {save_path}")


if __name__ == "__main__":
    main()
