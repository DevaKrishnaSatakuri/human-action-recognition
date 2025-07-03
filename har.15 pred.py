import os
import cv2
import joblib
import numpy as np
from skimage.feature import hog

# === CONFIG ===
model_path = "multiclass_hog_rf_model.pkl"
image_path = "watching_TV_194.jpgx"   # this to your test image path
resize_dim = (128, 128)

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'feature_vector': True
}

# === LOAD MODEL + LABEL ENCODER ===
model, label_encoder = joblib.load(model_path)

# === EXTRACT FEATURES ===
def extract_hog(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, resize_dim)
    return hog(img, **hog_params)

features = extract_hog(image_path).reshape(1, -1)

# === PREDICT ===
pred_class = model.predict(features)[0]
pred_label = label_encoder.inverse_transform([pred_class])[0]
print(f"\n🎯 Predicted Action: {pred_label}")
