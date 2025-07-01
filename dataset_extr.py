import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# ====== CONFIG ======
csv_path = "stanford40_full.csv"
image_dir = "JPEGImages"
resize_dim = (128, 128)
processed_img_dir = "processed_images"   # Save preprocessed images here
os.makedirs(processed_img_dir, exist_ok=True)

hog_params = {
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'orientations': 9,
    'block_norm': 'L2-Hys',
    'feature_vector': True
}
lbp_params = {'P': 8, 'R': 1}

# ====== Load CSV and Encode Labels ======
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X, y = [], []

# ====== Feature Extraction + Save Preprocessed Images ======
for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting & Saving"):
    img_path = os.path.join(image_dir, row['filename'])
    if not os.path.isfile(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    try:
        # Grayscale + Resize + Normalize
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize_dim)
        img_norm = img.astype(np.float32) / 255.0

        # === Save preprocessed image ===
        save_path = os.path.join(processed_img_dir, row['filename'])
        cv2.imwrite(save_path, (img_norm * 255).astype(np.uint8))  # Convert back to 0-255

        # Feature extraction
        hog_feat = hog(img_norm, **hog_params)
        lbp = local_binary_pattern(img_norm, **lbp_params)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)

        features = np.concatenate([hog_feat, lbp_hist])
        X.append(features)
        y.append(row['label_encoded'])
    except:
        continue

if len(X) == 0:
    raise ValueError("❌ No features extracted. Check image paths.")

# ====== Standardize + PCA ======
X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
X_pca = pca.fit_transform(X_scaled)

# ====== Save Features to CSV ======
df_features = pd.DataFrame(X_pca)
df_features['label'] = y
df_features.to_csv("har_features_pca.csv", index=False)

print("✅ Features saved to har_features_pca.csv")
print("✅ Preprocessed images saved to /content/processed_images")
