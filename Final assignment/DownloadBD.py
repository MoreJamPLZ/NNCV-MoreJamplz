from datasets import load_dataset
import os

# Using a community-uploaded version of BDD100K
# 'streaming=True' is the key—it doesn't download the whole dataset
ds = load_dataset("dgural/bdd100k", split="train", streaming=True)

os.makedirs("bdd100k_500", exist_ok=True)

# Take exactly 500 samples
for i, sample in enumerate(ds.take(500)):
    image = sample["image"]  # 'image' is a PIL object
    image.save(f"bdd100k_500/img_{i:03d}.jpg")

    if (i + 1) % 50 == 0:
        print(f"Downloaded {i + 1}/500...")

print("Done! Check the 'bdd100k_500' folder.")