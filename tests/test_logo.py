import torch
from torchvision import transforms
from PIL import Image
import logging
from src.logo import LogoRecognitionModel, LogoDet3KDataset

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, num_subcategories):
    model = LogoRecognitionModel(num_subcategories)
    model_dict = model.state_dict()  # Get the current model's state_dict
    
    # Load the pre-trained model weights
    pretrained_dict = torch.load(model_path)
    
    # Filter out layers with mismatched sizes (like the classifier layers)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].size() == v.size()}
    
    # Update the current model's state_dict with pre-trained weights
    model_dict.update(pretrained_dict)
    
    # Load the updated state_dict back into the model
    model.load_state_dict(model_dict)
    
    return model

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

        # Get the predicted subcategory and count
        subcategory_idx = torch.argmax(subcategory_output, dim=1).item()
        predicted_subcategory = subcategories[subcategory_idx]
        predicted_count = count_output.squeeze().item()

        return predicted_subcategory, predicted_count
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return None, None

def test_single_image(model_path, image_path, subcategories):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(model_path, len(subcategories))
    model.to(device)
    model.eval()

    predicted_subcategory, predicted_count = predict(model, image_path, subcategories, device)
    
    if predicted_subcategory and predicted_count is not None:
        logger.info(f"Image: {image_path}")
        logger.info(f"Predicted Logo (Subcategory): {predicted_subcategory}")
        logger.info(f"Predicted Count: {predicted_count:.2f}")
    else:
        logger.warning(f"Prediction failed for image: {image_path}")

def main():
    try:
        # Load the dataset to get the subcategories
        dataset = LogoDet3KDataset(root_dir='data/LogoDet-3K', transform=None, max_samples=1000)
        subcategories = dataset.subcategories

        model_path = 'models/logo_recognition_model.pth'
        
        # Test with hardcoded image path
        test_image_path = 'data/test/8.jpg'

        test_single_image(model_path, test_image_path, subcategories)

    except Exception as e:
        logger.error(f"An error occurred in main: {e}")

if __name__ == "__main__":
    main()
