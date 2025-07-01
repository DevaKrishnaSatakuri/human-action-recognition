import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# === CONFIG ===
csv_path = "stanford40_full.csv"
image_dir = "processed_images"
batch_size = 32
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === CUSTOM DATASET ===
class Grayscale3ChannelDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.le = LabelEncoder()
        self.data['label_encoded'] = self.le.fit_transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data.iloc[idx]['filename'])
        image = Image.open(img_path).convert("L")  # grayscale
        image = image.convert("RGB")  # convert to 3 channels (needed for MobileNet)
        label = self.data.iloc[idx]['label_encoded']
        if self.transform:
            image = self.transform(image)
        return image, label

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === LOAD DATA ===
dataset = Grayscale3ChannelDataset(csv_path, image_dir, transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# === LOAD PRETRAINED MOBILENET ===
model = models.mobilenet_v2(pretrained=True)
for param in model.features.parameters():
    param.requires_grad = False  # freeze feature extractor

# Replace classifier head
num_classes = len(dataset.le.classes_)
model.classifier[1] = nn.Linear(model.last_channel, num_classes)
model = model.to(device)

# === TRAINING ===
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🚀 Training MobileNetV2...")
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# === EVALUATION ===
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in dataloader:
        imgs = imgs.to(device)
        preds = model(imgs).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n📊 Accuracy:", accuracy_score(all_labels, all_preds))
print(classification_report(all_labels, all_preds, target_names=dataset.le.classes_))

# === SAVE MODEL ===
torch.save(model.state_dict(), "mobilenet_har_model.pth")
print("✅ Model saved as mobilenet_har_model.pth")
