import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from models.hybrid_model import HybridCNNViTModel
from PIL import Image
import numpy as np
from utils.image_processor import ImageProcessor

class PneumoniaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # Expect subfolders 'pneumonia' and 'normal' inside root_dir
        pneumonia_dir = os.path.join(root_dir, 'pneumonia')
        normal_dir = os.path.join(root_dir, 'normal')
        for fname in os.listdir(pneumonia_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                self.image_paths.append(os.path.join(pneumonia_dir, fname))
                self.labels.append(1)
        for fname in os.listdir(normal_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.dcm')):
                self.image_paths.append(os.path.join(normal_dir, fname))
                self.labels.append(0)
        self.image_processor = ImageProcessor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        # Load and preprocess image
        image = self.image_processor.preprocess(img_path)
        # Convert numpy array to tensor
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)  # C,H,W
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=10):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                # Adjust input channels if needed
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)  # Convert 1 channel to 3 channels by repeating
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
    return model

def main():
    data_dir = 'training_data'
    batch_size = 16
    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data augmentation transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Dataset and splits
    full_dataset = PneumoniaDataset(data_dir, transform=data_transforms['train'])
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Override val dataset transform
    val_dataset.dataset.transform = data_transforms['val']

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    }

    # Use hybrid CNN-ViT model
    model = HybridCNNViTModel(num_classes=2)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in dataloaders[phase]:
                # Adjust input channels if needed
                if inputs.shape[1] == 1:
                    inputs = inputs.repeat(1, 3, 1, 1)  # Convert 1 channel to 3 channels by repeating
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        if phase == 'train':
            scheduler.step()

    # Save the trained model
    torch.save(model.state_dict(), 'cnn_pneumonia_model.pth')
    print("Training complete and model saved as cnn_pneumonia_model.pth")

if __name__ == "__main__":
    main()
