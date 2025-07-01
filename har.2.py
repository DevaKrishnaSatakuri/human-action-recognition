import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torchvision import models, transforms

# ======== 1. Setup ============
image_dir = "JPEGImages"  # folder with all images
csv_file = "stanford40_full.csv"  # contains filename + label

df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Clean column names

# Encode labels
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# ======== 2. Pre-trained ResNet18 Model =============
resnet = models.resnet18(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer
resnet.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet.to(device)

# Transform for input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ======== 3. Feature Extraction ===========
features = []
labels = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    image_path = os.path.join(image_dir, row['filename'])
    label = row['label_encoded']

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = resnet(image_tensor).cpu().numpy().flatten()

        features.append(embedding)
        labels.append(label)

    except Exception as e:
        print(f"Error with {row['filename']}: {e}")

features = np.array(features)
labels = np.array(labels)

# ======== 4. Train-Test Split and SVM Training ===========
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels)

model = SVC(kernel='rbf', C=10, gamma='scale')
model.fit(X_train, y_train)

# ======== 5. Evaluation ===========
y_pred = model.predict(X_test)

print("\n✅ Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n✅ Classification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
