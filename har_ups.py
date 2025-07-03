import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

# === CONFIGURATION ===
csv_path = "balanced_upsampled_dataset.csv"
image_dir = "upsampled_images"
model_dir = "models_ups"
os.makedirs(model_dir, exist_ok=True)

resize_dim = (128, 128)  # Only needed if resizing again
hog_params = {
    'orientations': 9,
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'block_norm': 'L2-Hys',
    'feature_vector': True
}
models = {}
skipped_classes = []

# === LOAD DATA ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
all_labels = df['label'].unique()

# === HOG Feature Extractor
def extract_features(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return hog(img, **hog_params)

# === TRAINING LOOP ===
for action in all_labels:
    print(f"\n🔧 Training model for: {action}")
    pos_df = df[df['label'] == action]
    neg_df = df[df['label'] != action]

    if len(pos_df) < 10:
        print(f"⚠️ Skipping '{action}' — too few positive samples.")
        skipped_classes.append(action)
        continue

    combined_df = pd.concat([pos_df, neg_df]).sample(frac=1, random_state=42)

    X, y = [], []
    for _, row in tqdm(combined_df.iterrows(), total=len(combined_df), desc=f"Extracting: {action}"):
        img_path = os.path.join(image_dir, row['filename'])
        feat = extract_features(img_path)
        if feat is not None:
            X.append(feat)
            y.append(1 if row['label'] == action else 0)

    if len(X) < 10:
        print(f"⚠️ Skipping '{action}' — not enough usable features.")
        skipped_classes.append(action)
        continue

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    models[action] = clf

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ Accuracy for '{action}': {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not " + action, action]))

    # Save model
    model_path = os.path.join(model_dir, f"model_{action.replace(' ', '_')}.pkl")
    joblib.dump(clf, model_path)
    print(f"💾 Model saved to: {model_path}")

# === Summary
print(f"\n✅ Total models trained: {len(models)}")
if skipped_classes:
    print("\n⚠️ Skipped the following classes due to low or invalid data:")
    for cls in skipped_classes:
        print(f" - {cls}")
