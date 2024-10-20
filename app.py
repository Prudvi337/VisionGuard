from flask import Flask, request, jsonify, render_template
import torch 
from torchvision import transforms
from PIL import Image
import os
import io
import base64
from src.logo import LogoRecognitionModel
# Constants
IMG_SIZE = 128

# Data transform for freshness model
freshness_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to create the freshness model
def create_freshness_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 2)
    return model

# Function to create the logo recognition model
def create_logo_model(num_subcategories):
    model = LogoRecognitionModel(num_subcategories)
    return model

# Function to interpret freshness levels
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

# Function to make freshness predictions
def predict_freshness(image, model, device):
    model.eval()
    image_tensor = freshness_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        fresh_prob = probabilities[0].item()
        rotten_prob = probabilities[1].item()
        
    freshness_label = interpret_freshness(fresh_prob)
    
    return freshness_label, fresh_prob, rotten_prob

# Function to make logo predictions
def predict_logo(model, image_path, subcategories, device='cuda'):
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        subcategory_output, count_output = model(image_tensor)
    
    subcategory_idx = torch.argmax(subcategory_output, dim=1).item()
    predicted_subcategory = subcategories[subcategory_idx]
    predicted_count = count_output.squeeze().item()
    
    return predicted_subcategory, predicted_count

# Function to estimate shelf life
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

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
# Load the models
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load freshness model
freshness_model = create_freshness_model()
freshness_model.load_state_dict(torch.load('freshness_model.pth', map_location=device))
freshness_model.to(device)

# Load logo recognition model
# Assuming you have a way to get the number of subcategories and the subcategories list
num_subcategories = 153  # Update this with the correct number
subcategories = ['subcategory1', 'subcategory2', ...]  # Update this with the correct list
logo_model = create_logo_model(num_subcategories)
logo_model.load_state_dict(torch.load('logo_recognition_model.pth', map_location=device))
logo_model.to(device)

@app.route('/test_freshness', methods=['POST'])
def test_freshness():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    produce_type = request.form.get('produce_type', 'unknown')
    
    try:
        image = Image.open(image_file).convert('RGB')
        prediction, fresh_prob, rotten_prob = predict_freshness(image, freshness_model, device)
        shelf_life = estimate_shelf_life(prediction, fresh_prob, produce_type)
        
        return jsonify({
            'freshness': prediction,
            'fresh_probability': fresh_prob,
            'rotten_probability': rotten_prob,
            'estimated_shelf_life': shelf_life
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test_logo', methods=['POST'])
def test_logo():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    
    try:
        # Save the uploaded file temporarily
        temp_path = 'temp_image.jpg'
        image_file.save(temp_path)
        
        predicted_subcategory, predicted_count = predict_logo(logo_model, temp_path, subcategories, device)
        
        # Remove the temporary file
        os.remove(temp_path)
        
        return jsonify({
            'predicted_logo': predicted_subcategory,
            'predicted_count': predicted_count
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)