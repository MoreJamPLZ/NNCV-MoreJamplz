from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)
from model import Model

# Fixed paths inside participant container — do NOT change
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"

INPUT_SIZE = (512, 512)


def preprocess(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")

    transform = Compose([
        ToImage(),
        Resize(size=INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    img = transform(img)
    img = img.unsqueeze(0)  
    return img


def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    # Resize logits (still floats) to original shape before argmax
    pred = torch.nn.functional.interpolate(
        pred, size=original_shape, mode="bilinear", align_corners=False
    )
    prediction_numpy = torch.argmax(pred, dim=1).cpu().numpy().squeeze()
    return prediction_numpy


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Model()
    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True),
        strict=True,
    )
    model.eval().to(device)

    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path)
            original_shape = np.array(img).shape[:2]

            img_tensor = preprocess(img).to(device)
            seg_pred = model(img_tensor)
            seg_pred = postprocess(seg_pred, original_shape)

            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)

            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

    print(f"Saved {len(image_files)} predictions to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()