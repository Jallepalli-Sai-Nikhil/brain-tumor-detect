import os
import gdown
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # type: ignore

# Google Drive Model URL (replace this with your actual model file ID or URL)
MODEL_URL = "https://drive.google.com/file/d/1UukTHheHnqLZVX1u0wpI7fEjO6Y8-6MT/view?usp=sharing"

# Download Model if it doesn't exist locally
MODEL_PATH = "brain_tumor_model_xception.h5"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# Load Trained Model
model = tf.keras.models.load_model(MODEL_PATH)

# Define Image Size (same as training)
IMG_SIZE = (299, 299)

# Class Labels (Update with actual class names from training)
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

# Initialize Flask App
app = Flask(__name__)

def preprocess_image(image_path):
    """Load and preprocess an image for prediction."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route("/")
def home():
    """Render the HTML UI."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Handle image prediction requests."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = "uploaded_image.jpg"
    file.save(file_path)

    # Preprocess & Predict
    img_array = preprocess_image(file_path)
    predictions = model.predict(img_array)
    predicted_class = CLASS_LABELS[np.argmax(predictions)]
    confidence = float(np.max(predictions))

    # Remove uploaded file after prediction
    os.remove(file_path)

    return jsonify({"predicted_class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
