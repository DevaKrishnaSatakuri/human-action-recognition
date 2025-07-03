import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from skimage.feature import hog
import joblib

# === CONFIG ===
csv_path = "balanced_upsampled_dataset.csv"
image_dir = "upsampled_images"
model_path = "multiclass_hog_rf_model.pkl"

hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'feature_vector': True
}

# === LOAD CSV
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# === Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# === Feature Extraction
X, y = [], []
skipped = 0

print("📸 Extracting HOG features from images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    img_path = os.path.join(image_dir, row['filename'])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        skipped += 1
        continue
    feat = hog(img, **hog_params)
    X.append(feat)
    y.append(row['label_encoded'])

print(f"✅ Features extracted: {len(X)}, Skipped images: {skipped}")

X = np.array(X)
y = np.array(y)

# === Split & Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=150, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# === Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🎯 Overall Accuracy: {acc:.4f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Save Model and LabelEncoder
joblib.dump((model, le), model_path)
print(f"\n💾 Model + LabelEncoder saved to: {model_path}")
