import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
csv_path = "stanford40_full.csv"           # Your CSV with 'filename' and 'label'
image_dir = "JPEGImages"              # Folder where images like 108622.jpg are stored
resize_dim = (128, 128)

hog_params = {
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'orientations': 9,
    'block_norm': 'L2-Hys',
    'feature_vector': True
}
lbp_params = {'P': 8, 'R': 1}

# === LOAD CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X, y = [], []
missing = 0

print("📷 Extracting features...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(image_dir, row['filename'])

    if not os.path.isfile(path):
        print(f"❌ Missing file: {row['filename']}")
        missing += 1
        continue

    img = cv2.imread(path)
    if img is None:
        print(f"⚠️ Unreadable image: {row['filename']}")
        missing += 1
        continue

    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize_dim)

        hog_feat = hog(img, **hog_params)
        lbp = local_binary_pattern(img, **lbp_params)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)

        features = np.concatenate([hog_feat, lbp_hist])
        X.append(features)
        y.append(row['label_encoded'])

    except Exception as e:
        print(f"⚠️ Error with {row['filename']}: {e}")
        continue

if len(X) == 0:
    raise ValueError("❌ No features extracted. Check image paths and formats.")

X = np.array(X)
y = np.array(y)

print(f"✅ Finished feature extraction. Skipped {missing} files.")

# === SCALE + SPLIT ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# === GRIDSEARCHCV + SVM ===
print("🔍 Tuning SVM...")
param_grid = {
    'C': [1, 10],
    'gamma': ['scale', 0.01],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

# === EVALUATION ===
model = grid.best_estimator_
y_pred = model.predict(X_test)

print("\n✅ Best Parameters:", grid.best_params_)
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# === CONFUSION MATRIX HEATMAP ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

print("🖼️ Confusion matrix saved as confusion_matrix.png")
