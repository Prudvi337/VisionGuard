import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BrandDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.brands = []

        try:
            self.brands = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            if not self.brands:
                raise ValueError(f"No brand folders found in {root_dir}.")
            
            for brand in self.brands:
                brand_path = os.path.join(root_dir, brand)
                images = [f for f in os.listdir(brand_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                for img_file in images:
                    self.image_files.append((brand, os.path.join(brand_path, img_file)))
            
            if not self.image_files:
                raise ValueError(f"No valid image files found in the dataset.")
            
            logger.info(f"Found {len(self.brands)} brands and {len(self.image_files)} images.")
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        brand, img_path = self.image_files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            return image, self.brands.index(brand)
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return None

class BrandRecognitionModel(nn.Module):
    def __init__(self, num_brands):
        super(BrandRecognitionModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.last_channel

        # Freeze the feature layers to speed up training
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.base_model.classifier = nn.Identity()
        self.brand_classifier = nn.Linear(num_features, num_brands)

    def forward(self, x):
        features = self.base_model(x)
        brand_output = self.brand_classifier(features)
        return brand_output

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, device='cuda'):
    model.to(device)
    scaler = torch.cuda.amp.GradScaler()  # For mixed precision training
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            images, brands = batch
            images, brands = images.to(device), brands.to(device)
            
            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                brand_output = model(images)
                loss = criterion(brand_output, brands)

            # Backward pass with mixed precision
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch is None:
                    continue
                images, brands = batch
                images, brands = images.to(device), brands.to(device)
                
                with torch.cuda.amp.autocast():  # Mixed precision during validation
                    brand_output = model(images)
                    loss = criterion(brand_output, brands)
                    val_loss += loss.item()

                    _, predicted = torch.max(brand_output, 1)
                    correct_predictions += (predicted == brands).sum().item()
                    total_predictions += brands.size(0)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f} | Validation Accuracy: {accuracy:.4f}")

        torch.cuda.empty_cache()

    torch.save(model.state_dict(), 'brand_recognition_model.pth')
    logger.info("Model saved to brand_recognition_model.pth")

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        data_transforms = transforms.Compose([
            transforms.Resize((160, 160)),  # Reducing the input image size to speed up training
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = BrandDataset(root_dir='data/Logo/train', transform=data_transforms)
        val_dataset = BrandDataset(root_dir='data/Logo/test', transform=data_transforms)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1, pin_memory=True)  # Increased batch size, reduced num_workers
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1, pin_memory=True)

        model = BrandRecognitionModel(num_brands=len(train_dataset.brands))
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, device=device)

        # Test prediction
        test_image_path = 'data/test/cadbury.jpg'
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((160, 160)),  # Match the training image size
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(test_image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.cuda.amp.autocast():  # Use mixed precision for inference
                brand_output = model(image_tensor)

        brand_idx = torch.argmax(brand_output, dim=1).item()
        predicted_brand = train_dataset.brands[brand_idx]

        logger.info(f"Predicted Brand: {predicted_brand}")

    except Exception as e:
        logger.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
