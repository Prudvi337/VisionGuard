import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast, GradScaler

# Set random seed for reproducibility
torch.manual_seed(42)

# Define constants
IMG_SIZE = 128  # Reduced from 224
BATCH_SIZE = 64  # Increased from 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5  # For early stopping

# Custom Dataset class (unchanged)
class FruitVegDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['fresh', 'rotten']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._make_dataset()

    def _make_dataset(self):
        samples = []
        for category in ['fruits', 'vegetables']:
            category_path = os.path.join(self.root_dir, category)
            for freshness in self.classes:
                class_path = os.path.join(category_path, freshness)
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    samples.append((img_path, self.class_to_idx[freshness]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Data transforms
data_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Model definition
def create_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model

# Training function
def train_model(model, criterion, optimizer, scheduler, dataloader, device, num_epochs):
    scaler = GradScaler()
    best_acc = 0.0
    patience_counter = 0
    train_losses = []
    train_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

        scheduler.step()

        # Early stopping
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'freshness_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return train_losses, train_accuracies

# Main function
def main():
    # Create dataset and dataloader
    dataset = FruitVegDataset(root_dir='data/Fruits_Vegetables_Dataset(12000)', transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create model, loss function, and optimizer
    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    train_losses, train_accuracies = train_model(model, criterion, optimizer, scheduler, dataloader, device, EPOCHS)

    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    print("Training completed.")

if __name__ == '__main__':
    main()
