import os
import cv2
import pandas as pd
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load CSV
df = pd.read_csv("stanford40_openml.csv")  # Columns: filename, label

# Image directory
image_dir = "JPEGImages"
# Preprocessing + feature extraction
X = []
y = []

for idx, row in df.iterrows():
    path = os.path.join(image_dir, row['filename'])
    label = row['label']

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (128, 128))  # Resizing helps HOG

    features = hog(img, pixels_per_cell=(8,8), cells_per_block=(2, 2),block_norm='L2-Hys',visualize=False, feature_vector=True)
    X.append(features)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
model = SVC(kernel='linear', C=1,class_weight='balanced')
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))