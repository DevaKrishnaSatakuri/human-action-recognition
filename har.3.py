import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage import exposure
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# ===== CONFIG =====
image_dir = "JPEGImages"
csv_file = "stanford40_full.csv"
resize_dim = (128, 128)

hog_params = {
    'pixels_per_cell': (8, 8),
    'cells_per_block': (2, 2),
    'orientations': 9,
    'block_norm': 'L2-Hys',
    'feature_vector': True
}
lbp_params = {'P': 8, 'R': 1}

# ===== LOAD LABELS =====
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = []
y = []

# ===== VISUALIZATION (Optional: Show first image HOG + LBP) =====
def visualize_hog_lbp(img, title=""):
    # HOG
    _, hog_image = hog(img, visualize=True, **hog_params)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # LBP
    lbp = local_binary_pattern(img, **lbp_params)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(hog_image, cmap='gray')
    axs[1].set_title('HOG')
    axs[2].imshow(lbp, cmap='gray')
    axs[2].set_title('LBP')
    for ax in axs:
        ax.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ===== FEATURE EXTRACTION =====
for i, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join(image_dir, row['filename'])
    label = row['label_encoded']

    try:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, resize_dim)

        # Visualize first image only
        if i == 0:
            visualize_hog_lbp(img, title=f"Sample: {row['label']}")

        hog_feat = hog(img, **hog_params)
        lbp = local_binary_pattern(img, **lbp_params)
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), density=True)

        features = np.concatenate([hog_feat, lbp_hist])
        X.append(features)
        y.append(label)

    except Exception as e:
        print(f"Error processing {row['filename']}: {e}")

X = np.array(X)
y = np.array(y)

# ===== SCALING + SPLIT =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ===== GRIDSEARCHCV FOR SVM =====
param_grid = {
    'C': [1, 10, 100],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(class_weight='balanced'), param_grid, cv=3, verbose=2, n_jobs=-1)
grid.fit(X_train, y_train)

print("\n✅ Best Parameters:", grid.best_params_)

# ===== EVALUATION =====
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("\n✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

# ===== CONFUSION MATRIX HEATMAP =====
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()
