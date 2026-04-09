"""
Package model.pt for OOD submission.
Combines:
  1. SegFormer-B5 pretrained weights (from your checkpoint)
  2. DINOv3 pretrained weights (from .pth file)
  3. Nonlinear CNN novice weights (randomly initialized with seed)

Run this ONCE to create model.pt, then build Docker image.

Usage:
    python package_model.py
"""

import random
import numpy as np
import torch
from model import Model

# ============================================================================
# Configuration — EDIT THESE PATHS
# ============================================================================
SEED = 1
SEGFORMER_CHECKPOINT = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/segformer.pt"  # your pretrained SegFormer weights
DINOV3_WEIGHTS = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/dinov3_vitb16_timm.pth"
OUTPUT_PATH = "model.pt"

# OOD threshold from ablation 3 (Nonlinear CNN, median, n=20)
OOD_THRESHOLD = 5.4385


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    print("Building model...")
    set_seed(SEED)
    model = Model()

    # 1. Load SegFormer pretrained weights
    print(f"Loading SegFormer weights from: {SEGFORMER_CHECKPOINT}")
    seg_state = torch.load(SEGFORMER_CHECKPOINT, map_location="cpu", weights_only=True)
    # The checkpoint might have full Model state_dict or just segformer
    # Try to load segformer keys specifically
    seg_keys = {k: v for k, v in seg_state.items() if k.startswith("segformer.")}
    if seg_keys:
        model.load_state_dict(seg_keys, strict=False)
        print(f"  Loaded {len(seg_keys)} SegFormer keys")
    else:
        # Might be a raw SegFormer state_dict without prefix
        prefixed = {f"segformer.{k}": v for k, v in seg_state.items()}
        model.load_state_dict(prefixed, strict=False)
        print(f"  Loaded {len(prefixed)} SegFormer keys (added prefix)")

    # 2. Load DINOv3 weights
    print(f"Loading DINOv3 weights from: {DINOV3_WEIGHTS}")
    dino_state = torch.load(DINOV3_WEIGHTS, map_location="cpu", weights_only=True)
    if "model" in dino_state:
        dino_state = dino_state["model"]
    prefixed_dino = {f"dino.{k}": v for k, v in dino_state.items()}
    model.load_state_dict(prefixed_dino, strict=False)
    print(f"  Loaded {len(prefixed_dino)} DINOv3 keys")

    # 3. Novice is already randomly initialized (seed was set before Model())
    novice_params = sum(p.numel() for p in model.novice.parameters())
    print(f"  Novice (random init, seed={SEED}): {novice_params:,} parameters")

    # 4. Set threshold
    model.ood_threshold.fill_(OOD_THRESHOLD)
    print(f"  OOD threshold: {OOD_THRESHOLD}")

    # 5. Save complete state_dict
    torch.save(model.state_dict(), OUTPUT_PATH)
    print(f"\nSaved → {OUTPUT_PATH}")

    # 6. Verify round-trip
    print("\nVerifying round-trip load...")
    model2 = Model()
    model2.load_state_dict(torch.load(OUTPUT_PATH, map_location="cpu", weights_only=True), strict=True)
    print("  strict=True load succeeded!")

    # Quick sanity check
    x = torch.randn(1, 3, 512, 512)
    model2.eval()
    with torch.no_grad():
        logits, include = model2(x)
    print(f"  Test forward: logits={logits.shape}, include={include}")
    print("\nDone! Copy model.pt to 'Final assignment/' and build Docker image.")


if __name__ == "__main__":
    main()