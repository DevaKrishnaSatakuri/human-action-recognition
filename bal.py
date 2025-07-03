import os
import pandas as pd
import shutil
from sklearn.utils import resample

# === INPUT PATHS ===
csv_path = "stanford40_full.csv"
image_dir = "processed_images"
output_dir = "upsampled_images"
output_csv = "balanced_upsampled_dataset.csv"

# === LOAD AND CLEAN CSV ===
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

# Remove entries without actual image file
df['exists'] = df['filename'].apply(lambda x: os.path.isfile(os.path.join(image_dir, x)))
df = df[df['exists']]
df.drop(columns='exists', inplace=True)

# === UPSAMPLE ===
max_samples = df['label'].value_counts().max()
print(f"📊 Upsampling all classes to {max_samples} samples...")

upsampled_df = []

for label, group in df.groupby('label'):
    if len(group) < max_samples:
        upsampled = resample(group, replace=True, n_samples=max_samples, random_state=42)
    else:
        upsampled = group
    upsampled_df.append(upsampled)

final_df = pd.concat(upsampled_df).reset_index(drop=True)

# === SAVE UPSAMPLED CSV ===
final_df.to_csv(output_csv, index=False)
print(f"\n✅ Upsampled CSV saved to '{output_csv}'")

# === OPTIONAL: COPY IMAGES INCLUDING DUPLICATES ===
os.makedirs(output_dir, exist_ok=True)
copied = 0

for i, row in final_df.iterrows():
    src = os.path.join(image_dir, row['filename'])

    # To avoid name conflict, rename duplicate files with index
    dst_filename = f"{i:05d}_{row['filename']}"
    dst = os.path.join(output_dir, dst_filename)

    try:
        shutil.copy2(src, dst)
        final_df.loc[i, 'filename'] = dst_filename  # update filename in CSV
        copied += 1
    except Exception as e:
        print(f"⚠️ Error copying {row['filename']}: {e}")

# === SAVE FINAL CSV WITH UPDATED FILENAMES ===
final_df.to_csv(output_csv, index=False)
print(f"✅ Copied {copied} images to '{output_dir}'")
print("\n📦 Final class distribution:")
print(final_df['label'].value_counts())
