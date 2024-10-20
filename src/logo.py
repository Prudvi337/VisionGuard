import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogoDet3KDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_xml_pairs = []
        self.categories = []
        self.subcategories = []

        try:
            self.categories = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            if not self.categories:
                raise ValueError(f"No category folders found in {root_dir}.")
            
            for category in self.categories:
                category_path = os.path.join(root_dir, category)
                brands = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
                
                for brand in brands:
                    if brand not in self.subcategories:
                        self.subcategories.append(brand)
                    brand_path = os.path.join(category_path, brand)
                    image_files = [f for f in os.listdir(brand_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                    
                    for img_file in image_files:
                        xml_file = os.path.splitext(img_file)[0] + '.xml'
                        if os.path.exists(os.path.join(brand_path, xml_file)):
                            self.image_xml_pairs.append((category, brand, img_file, xml_file))
                            if max_samples and len(self.image_xml_pairs) >= max_samples:
                                break
                    if max_samples and len(self.image_xml_pairs) >= max_samples:
                        break
                if max_samples and len(self.image_xml_pairs) >= max_samples:
                    break
            
            if not self.image_xml_pairs:
                raise ValueError(f"No valid image-XML pairs found in the dataset.")
            
            logger.info(f"Found {len(self.categories)} categories, {len(self.subcategories)} subcategories, and {len(self.image_xml_pairs)} image-XML pairs.")
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise

    def __len__(self):
        return len(self.image_xml_pairs)

    def __getitem__(self, idx):
        category, brand, img_file, xml_file = self.image_xml_pairs[idx]
        img_path = os.path.join(self.root_dir, category, brand, img_file)
        xml_path = os.path.join(self.root_dir, category, brand, xml_file)

        try:
            image = Image.open(img_path).convert('RGB')
            
            tree = ET.parse(xml_path)
            root = tree.getroot()
            object_count = len(root.findall('object'))

            if self.transform:
                image = self.transform(image)

            return image, self.subcategories.index(brand), object_count, category
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            return None

class LogoRecognitionModel(nn.Module):
    def __init__(self, num_subcategories):
        super(LogoRecognitionModel, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        num_features = self.base_model.last_channel
        self.base_model.classifier = nn.Identity()
        self.subcategory_classifier = nn.Linear(num_features, num_subcategories)
        self.count_regressor = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.base_model(x)
        subcategory_output = self.subcategory_classifier(features)
        count_output = self.count_regressor(features)
        return subcategory_output, count_output

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, device='cuda'):
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            if batch is None:
                continue
            images, subcategories, counts, _ = batch
            images, subcategories, counts = images.to(device), subcategories.to(device), counts.float().to(device)
            
            optimizer.zero_grad()
            subcategory_output, count_output = model(images)
            
            loss_subcategory = criterion['subcategory'](subcategory_output, subcategories)
            loss_count = criterion['count'](count_output.squeeze(), counts)
            
            total_loss = loss_subcategory + loss_count
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} | Training Loss: {running_loss/len(train_loader):.4f}")

        # Validation at the end of each epoch
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch is None:
                    continue
                images, subcategories, counts, _ = batch
                images, subcategories, counts = images.to(device), subcategories.to(device), counts.float().to(device)
                
                subcategory_output, count_output = model(images)
                
                loss_subcategory = criterion['subcategory'](subcategory_output, subcategories)
                loss_count = criterion['count'](count_output.squeeze(), counts)
                val_loss += (loss_subcategory + loss_count).item()

                _, predicted = torch.max(subcategory_output, 1)
                correct_predictions += (predicted == subcategories).sum().item()
                total_predictions += subcategories.size(0)

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        logger.info(f"Validation Loss: {val_loss/len(val_loader):.4f} | Validation Accuracy: {accuracy:.4f}")

        # Clear cache
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), 'models/logo_recognition_model.pth')
    logger.info("Model saved to logo_recognition_model.pth")

def predict(model, image_path, subcategories, device='cuda'):
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            subcategory_output, count_output = model(image_tensor)

        subcategory_idx = torch.argmax(subcategory_output, dim=1).item()
        predicted_subcategory = subcategories[subcategory_idx]
        predicted_count = count_output.squeeze().item()

        return predicted_subcategory, predicted_count
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, None

def main():
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Limit the dataset size
        max_samples = 10000  # Adjust this number based on your available memory
        dataset = LogoDet3KDataset(root_dir='data/LogoDet-3K', transform=data_transforms, max_samples=max_samples)
        train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

        train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_data, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

        model = LogoRecognitionModel(num_subcategories=len(dataset.subcategories))
        criterion = {
            'subcategory': nn.CrossEntropyLoss(),
            'count': nn.MSELoss()
        }
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3, device=device)

        # Test prediction
        test_image_path = 'data/test/cadbury.jpg'
        predicted_subcategory, predicted_count = predict(model, test_image_path, dataset.subcategories, device=device)
        
        if predicted_subcategory and predicted_count:
            logger.info(f"Predicted Logo (Subcategory): {predicted_subcategory}, Predicted Count: {predicted_count:.2f}")
        else:
            logger.warning("Prediction failed.")

    except Exception as e:
        logger.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()