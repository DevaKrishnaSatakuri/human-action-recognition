import cv2
import joblib
import numpy as np
from skimage.feature import hog

model_path = "multiclass_hog_rf_cld_model.pkl"
image_path = "climbing_118.jpg"
resize_dim = (128, 128)

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'feature_vector': True
}

model, label_encoder = joblib.load(model_path)

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, resize_dim)
features = hog(img, **hog_params).reshape(1, -1)

prediction = model.predict(features)[0]
label = label_encoder.inverse_transform([prediction])[0]

print(f"\nPredicted Action: {label}")
