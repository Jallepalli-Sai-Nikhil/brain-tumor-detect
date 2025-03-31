import os
import gdown
import numpy as np
import tensorflow as tf
import logging
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MODEL_ID = "1UukTHheHnqLZVX1u0wpI7fEjO6Y8-6MT"  # Replace with actual Google Drive file ID
MODEL_PATH = "brain_tumor_model_xception.h5"
IMG_SIZE = (299, 299)  # Model's expected input size
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
UPLOAD_FOLDER = "uploads"

# Class Labels (Update based on training data)
CLASS_LABELS = ["Glioma Tumor", "Meningioma Tumor", "Pituitary Tumor", "No Tumor"]

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Download Model if not present
if not os.path.exists(MODEL_PATH):
    logger.info("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={MODEL_ID}", MODEL_PATH, quiet=False)

# Load Model with Error Handling
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading model: {e}")
    model = None

# Initialize Flask App
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def preprocess_image(image_path):
    """Load and preprocess an image for model prediction."""
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        return None

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

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    # Secure filename and save temporarily
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], "uploaded_image.jpg")
    file.save(file_path)

    # Preprocess and Predict
    img_array = preprocess_image(file_path)
    if img_array is None:
        os.remove(file_path)
        return jsonify({"error": "Error processing image"}), 500

    try:
        predictions = model.predict(img_array)
        predicted_class = CLASS_LABELS[np.argmax(predictions)]
        confidence = float(np.max(predictions))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        os.remove(file_path)
        return jsonify({"error": "Model prediction failed"}), 500

    # Cleanup: Remove uploaded file
    os.remove(file_path)

    return jsonify({"predicted_class": predicted_class, "confidence": confidence})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
