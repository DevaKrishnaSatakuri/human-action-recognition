import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

# === Config ===
model_dir = "models"
image_path = "applauding_207.jpg"

resize_dim = (128, 128)
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'feature_vector': True
}

# === Extract features from the image
def extract_features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {path}")
    img = cv2.resize(img, resize_dim)
    return hog(img, **hog_params)

# === Load all models
models = {}
for fname in os.listdir(model_dir):
    if fname.endswith(".pkl"):
        label = fname.replace("model_", "").replace(".pkl", "").replace("_", " ")
        models[label] = joblib.load(os.path.join(model_dir, fname))

print(f"✅ Loaded {len(models)} models.")

# === Predict
features = extract_features(image_path)
predictions = {}

for label, model in models.items():
    prob = model.predict_proba([features])[0][1]
    predictions[label] = prob

# === Top 5 Predictions
top = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5]
print("\n🎯 Top Predictions:")
for action, prob in top:
    print(f"{action}: {prob:.2f}")
