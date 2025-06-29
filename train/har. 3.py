import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ======== CONFIG =========
image_dir = "train"
csv_path = "C:\Users\PRAVALLIKA\IdeaProjects\har\Training_set.csv"
resize_dim = (128, 128)
hog_params = {
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'orientations': 9,
    'block_norm': 'L2-Hys',
    'feature_vector': True
}
lbp_params = {'P': 8, 'R': 1}

# ======== LOAD & ENCODE DATA =========
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = []
y = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(image_dir, row['filename'])
    label = row['label_encoded']

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, resize_dim)

    hog_features = hog(img, **hog_params)
    lbp = local_binary_pattern(img, **lbp_params)
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)

    features = np.concatenate([hog_features, lbp_hist])
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# ======== PREPROCESSING =========
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ======== GRIDSEARCH ON RANDOM FOREST =========
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 20, 40],  # deeper trees ~ more complex, like gamma
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42)

grid = GridSearchCV(rf, param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print("\n✅ Best Parameters Found:")
print(grid.best_params_)

# ======== EVALUATION =========
best_rf = grid.best_estimator_
y_pred = best_rf.predict(X_test)

print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
