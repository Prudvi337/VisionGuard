# VisionGuard

VisionGuard is an innovative AI-driven solution aimed at improving quality control in the e-commerce and food industries. By combining logo recognition and freshness detection models, VisionGuard enhances inventory tracking, reduces food waste, and promotes sustainability.

## Key Features
- **Logo Recognition**: Utilizes Optical Character Recognition (OCR) to extract brand details from packaging and Machine Learning (ML) for accurate logo detection.
- **Freshness Detection**: Assesses the freshness of produce by analyzing visual cues to predict shelf life.
- **Infrared Product Counting**: Integrates infrared technology for accurate item counting, helping to track inventory more efficiently.
- **User-Friendly Interface**: Provides a seamless drag-and-drop image upload system with real-time analysis and results.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Models](#models)
  - [Logo Recognition](#logo-recognition)
  - [Freshness Detection](#freshness-detection)
- [Future Scope](#future-scope)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run VisionGuard locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/VisionGuard.git
   ```
Navigate to the project directory:

bash
Copy code
cd VisionGuard
Set up the required dependencies by installing them via pip:

bash
Copy code
pip install -r requirements.txt
Download the pre-trained models for logo recognition and freshness detection:

logo_recognition_model.pth
freshness_model.pth
Run the application:

bash
Copy code
python app.py
## Usage
- **Image Input**: Upload images through the user interface (drag-and-drop or browse).
- **Choose Model**: Toggle between the Logo Recognition and Freshness Detection models.
- **Analyze**: Click on the "Analyze" button to initiate the process.
- **Results**: View the output with predicted brand names (logo recognition) or freshness status (freshness detection) along with their respective confidence scores.

## Project Workflow
1. **Image Upload**: Users can upload an image containing a product.
2. **Model Selection**:
   - **Logo Recognition**:
     - OCR extracts text from packaging.
     - The ML model detects the brand using a pre-trained neural network.
   - **Freshness Detection**:
     - The system analyzes the visual cues of fresh produce to predict shelf life.
3. **Display Results**: Outputs the predicted brand name and freshness status, each with a confidence score.

## Technologies Used
- **Python**: Core programming language.
- **Flask**: Web framework for the application.
- **OpenCV**: Image processing.
- **PyTorch**: Machine learning framework for building models.
- **OCR**: Used for extracting text from packaging.
- **Infrared Technology**: Integrated for product counting.
- **HTML/CSS/JavaScript**: For front-end development and user interface.

## Models
### Logo Recognition
- **Model**: Uses transfer learning and OCR to recognize and predict brand logos.
- **Data**: Trained on the LogoDet-3K dataset.

### Freshness Detection
- **Model**: Optimized for mobile devices, this model distinguishes between fresh and rotten produce by analyzing visual data.
- **Data Augmentation**: Employed to improve model generalization across different types of produce images.

## Future Scope
- Expand VisionGuard to include more product categories and retail industries.
- Establish partnerships with retailers to integrate into their inventory management systems.
- Continually refine models with additional data and user feedback.

## Contributing
We welcome contributions to VisionGuard! Please follow these steps:

1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-branch
Commit your changes:

bash
Copy code
git commit -m 'Add feature'
Push to your branch:

bash
Copy code
git push origin feature-branch
Open a pull request, and we’ll review your code.

sql
Copy code

This should match the formatting style of the first part. Let me know if you need any more adjustments!
