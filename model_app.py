import torch
import segmentation_models_pytorch as smp
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
import numpy as np
import base64

app = Flask(__name__)

# Load the model
def load_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    #model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))  # Load your saved model
    model.eval()
    return model

model = load_model()

# Preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))
    image = np.array(image).astype('float32') / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change shape to (C, H, W)
    image = torch.tensor(image).unsqueeze(0)  # Add batch dimension
    return image

# Convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).squeeze(0)  # Convert shape from (1, 1, H, W) to (H, W)
    image = tensor.cpu().numpy()
    image = (image * 255).astype(np.uint8)  # Scale to 0-255 for image representation
    return Image.fromarray(image, mode='L')  # 'L' mode for grayscale image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Open image and preprocess it
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        input_tensor = preprocess_image(image)

        # Perform inference
        with torch.no_grad():
            output = model(input_tensor)

        # Convert output to image
        output_image = tensor_to_image(output)
        
        # Convert both images to base64 for display in the HTML
        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        original_image_b64 = base64.b64encode(image_buffer.getvalue()).decode('utf-8')
        
        output_buffer = io.BytesIO()
        output_image.save(output_buffer, format="PNG")
        output_image_b64 = base64.b64encode(output_buffer.getvalue()).decode('utf-8')

        return jsonify({'original_image': original_image_b64, 'output_image': output_image_b64})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6009)
