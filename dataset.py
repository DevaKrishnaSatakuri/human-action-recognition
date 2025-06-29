import os
import pandas as pd

image_dir = "JPEGImages"
data = []

for fname in os.listdir(image_dir):
    if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
        label = fname.split('_')[0].lower()
        data.append([fname, label])

df = pd.DataFrame(data, columns=['filename', 'label'])
df.to_csv("stanford40_full.csv", index=False)
print("✅ Full CSV created with", len(df), "entries.")
