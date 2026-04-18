"""
Prediction pipeline for OOD-aware segmentation submission.
Outputs segmentation masks + predictions.csv with include decisions.
"""
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode,
)

from model import Model
import os
import csv

IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"


def preprocess(img: Image.Image) -> torch.Tensor:
    transform = Compose([
        ToImage(),
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return transform(img).unsqueeze(0)


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    pred_upsampled = nn.functional.interpolate(
        pred, size=original_shape, mode="bilinear", align_corners=False,
    )
    pred_max = torch.argmax(pred_upsampled, dim=1)
    return pred_max.cpu().detach().numpy().squeeze()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval().to(device)
    print(f"OOD method: {model.ood_method}")
    print(f"OOD threshold: {model._active_threshold():.4f}")

    image_files = []
    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(Path(IMAGE_DIR).rglob(ext))
    print(f"Found {len(image_files)} images to process.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    predictions = []

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            seg_pred, include_decision = model(img_tensor)

            seg_pred = postprocess(seg_pred, original_shape)

            relative_path = img_path.relative_to(IMAGE_DIR)
            out_path = Path(OUTPUT_DIR) / relative_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

            predictions.append({
                "image_name": str(relative_path).replace("\\", "/"),
                "include": bool(include_decision),
            })

            del img_tensor, seg_pred

    csv_path = Path(OUTPUT_DIR) / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "include"])
        writer.writeheader()
        writer.writerows(predictions)

    print(f"Saved {len(predictions)} predictions to {csv_path}")


if __name__ == "__main__":
    main()