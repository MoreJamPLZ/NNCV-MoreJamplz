
"""
Prediction pipeline for SegFormer-B5 semantic segmentation on Cityscapes.
Loads a pre-trained model, processes input images, and saves predicted segmentation masks.
 
Compatible with the challenge submission server.
"""
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
 
# Fixed paths inside participant container — DO NOT CHANGE
IMAGE_DIR = "/data"
OUTPUT_DIR = "/output"
MODEL_PATH = "/app/model.pt"
 
 
def preprocess(img: Image.Image) -> torch.Tensor:
    """
    Preprocess a PIL image for SegFormer-B5.
    Uses 1024x1024 to match the pretrained model's training resolution.
    Normalization uses ImageNet statistics (what SegFormer was trained with).
    """
    transform = Compose([
        ToImage(),
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    return img
 
 
def postprocess(pred: torch.Tensor, original_shape: tuple) -> np.ndarray:
    """
    Postprocess model output to a segmentation mask at original resolution.
    
    SegFormer outputs logits at 1/4 input resolution, so we upsample
    back to the original image size.
    """
    # Upsample logits to original resolution
    pred_upsampled = nn.functional.interpolate(
        pred, size=original_shape, mode="bilinear", align_corners=False
    )
    # Get class predictions
    pred_max = torch.argmax(pred_upsampled, dim=1)  # (B, H, W)
    prediction_numpy = pred_max.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()  # Remove batch dim
    return prediction_numpy
 
 
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
 
    # Load model
    model = Model()
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        state_dict,
        strict=True,
    )
    model.eval().to(device)
 
    image_files = list(Path(IMAGE_DIR).glob("*.png"))
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            original_shape = np.array(img).shape[:2]  # (H, W)

            # Preprocess
            img_tensor = preprocess(img).to(device)

            # FIX 1: Use Automatic Mixed Precision (AMP) to halve memory usage
            with torch.autocast(device_type="cuda" if "cuda" in device else "cpu"):
                pred = model(img_tensor)

            # Convert back to standard precision for the upsampling math
            pred = pred.float()

            # Postprocess to segmentation mask
            seg_pred = postprocess(pred, original_shape)

            # Save predicted mask
            out_path = Path(OUTPUT_DIR) / img_path.name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(seg_pred.astype(np.uint8)).save(out_path)

            # FIX 2: Explicitly delete large tensors to free GPU memory immediately
            del img_tensor, pred

    print("Prediction complete!")

 
 
if __name__ == "__main__":
    main()