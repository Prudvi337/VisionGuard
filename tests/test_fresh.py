import torch 
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Define constants
IMG_SIZE = 128

# Define the data transform (same as used in training)
data_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to create the model (same as in training script)
def create_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    return model

# Function to interpret freshness levels based on model's softmax output
def interpret_freshness(fresh_prob):
    if fresh_prob > 0.9:
        return "super fresh"
    elif fresh_prob > 0.7:
        return "fresh"
    elif fresh_prob > 0.5:
        return "slightly fresh"
    elif fresh_prob > 0.3:
        return "slightly rotten"
    elif fresh_prob > 0.1:
        return "rotten"
    else:
        return "very rotten"

# Function to make predictions
def predict_image(image_path, model, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image_tensor = data_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]  # Apply softmax to get probabilities
        fresh_prob = probabilities[0].item()  # Probability for 'fresh'
        rotten_prob = probabilities[1].item()  # Probability for 'rotten'
        
    # Interpret the freshness based on the 'fresh' probability
    freshness_label = interpret_freshness(fresh_prob)
    
    return freshness_label, fresh_prob, rotten_prob

# Function to estimate shelf life based on freshness and produce type
def estimate_shelf_life(freshness, fresh_probability, produce_type):
    base_shelf_life = {
        'apple': 14,
        'mango': 7,
        'strawberry': 4,
        'banana': 5,
        'carrot': 21,
        'cucumber': 10,
        'pepper': 7,
        'tomato': 7,
        'potato': 30,
    }

    base_life = base_shelf_life.get(produce_type.lower(), 7)  # Default to 7 if not found

    if freshness == 'fresh':
        adjusted_life = base_life * fresh_probability
    else:
        adjusted_life = base_life * (fresh_probability / 3)

    adjusted_life = max(1, round(adjusted_life, 1))
    return adjusted_life

# Function to display image with prediction and estimated shelf life
def display_prediction(image_path, prediction, fresh_prob, rotten_prob, shelf_life):
    img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {prediction} (Fresh: {fresh_prob:.2f}, Rotten: {rotten_prob:.2f})\nEstimated Shelf Life: {shelf_life} days")
    plt.show()

# Main function to load model and test images
def main():
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model = create_model()
    model.load_state_dict(torch.load('freshness_model.pth', map_location=device))
    model.to(device)
    print("Model loaded successfully.")

    # List of test images
    test_images = [
        'tests/strawberry.jpg',
    ]

    # Test the model with multiple images
    for img_path in test_images:
        prediction, fresh_prob, rotten_prob = predict_image(img_path, model, device)
        
        # Determine produce type from filename
        produce_type = os.path.basename(img_path).split('_')[0]  # e.g., "apple_01.jpg" -> "apple"
        shelf_life = estimate_shelf_life(prediction, fresh_prob, produce_type)
        
        print(f"The image {img_path} is predicted to be: {prediction} (Fresh: {fresh_prob:.2f}, Rotten: {rotten_prob:.2f})")
        print(f"Estimated shelf life: {shelf_life} days")
        display_prediction(img_path, prediction, fresh_prob, rotten_prob, shelf_life)

if __name__ == '__main__':
    main()
