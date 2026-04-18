"""
Threshold tuning script for logit-based OOD detection baselines.

Dataset setup:
  - FishyScapes  : tune/val  — RGB and mask have IDENTICAL filenames, different folders
  - Cityscapes   : tune/val  — clean street scenes, all in-distribution
  - RoadAnomaly21: test only — all images are OOD by dataset definition

Run:
    python tune_threshold.py --method energy
    python tune_threshold.py --method maxlogit
    python tune_threshold.py --method entropy
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode,
)
from sklearn.metrics import roc_auc_score
from model import Model

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE = Path("/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz")

MODEL_PATH  = BASE / "Final assignment/model.pt"

FISHY_RGB   = BASE / "fishyscapes_rgb_100"        # RGB images
FISHY_MASKS = BASE / "fishyscapes_lostandfound"   # masks — SAME filenames as RGB

CITY_VAL    = BASE / "data/cityscapes/leftImg8bit/val"
N_CITY      = 100

ROAD_RGB    = BASE / "dataset_AnomalyTrack/images"  # all OOD by definition

# ── Device ─────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(img: Image.Image) -> torch.Tensor:
    t = Compose([
        ToImage(),
        Resize((512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return t(img).unsqueeze(0)

# ── Scoring ───────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=["energy", "maxlogit", "entropy"],
        default="energy",
        help="OOD score computed from SegFormer logits.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to checkpoint (.pt). Can be full Model state_dict or SegFormer-only state_dict.",
    )
    return parser.parse_args()


def load_weights(model: Model, ckpt_path: Path, device: str):
    if not ckpt_path.exists():
        local_pt = sorted(BASE.rglob("*.pt"))
        hint = "\n".join(f"  - {p}" for p in local_pt[:10])
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            f"Found these .pt files:\n{hint if hint else '  (none)'}"
        )

    sd = torch.load(ckpt_path, map_location=device, weights_only=True)

    # 1) Try full strict load first (submission packaged model.pt case).
    try:
        model.load_state_dict(sd, strict=True)
        print("Loaded full model checkpoint (strict=True).")
        return
    except RuntimeError:
        pass

    # 2) Try loading segformer-prefixed keys only.
    if isinstance(sd, dict):
        seg_keys = {k: v for k, v in sd.items() if k.startswith("segformer.")}
        if seg_keys:
            model.load_state_dict(seg_keys, strict=False)
            print(f"Loaded SegFormer-prefixed checkpoint keys: {len(seg_keys)}")
            return

        # 3) Try raw SegFormer state_dict (no segformer. prefix).
        prefixed = {f"segformer.{k}": v for k, v in sd.items()}
        model.load_state_dict(prefixed, strict=False)
        print(f"Loaded raw SegFormer checkpoint keys (prefixed): {len(prefixed)}")
        return

    raise RuntimeError("Unsupported checkpoint format for loading model weights.")


@torch.no_grad()
def image_score(model, img_path: Path, device: str, method: str) -> float:
    img    = Image.open(img_path).convert("RGB")
    x      = preprocess(img).to(device)
    out    = model.segformer(pixel_values=x)
    logits = out.logits.float()

    if method == "energy":
        return float((-torch.logsumexp(logits, dim=1)).mean().item())

    if method == "maxlogit":
        return float((-logits.amax(dim=1)).mean().item())

    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return float(entropy.mean().item())

# ── Mask interpretation ────────────────────────────────────────────────────────
def mask_has_ood(mask_path: Path) -> bool:
    """
    Returns True if the mask contains any OOD pixels.
    FishyScapes uses: 1 = OOD object, 0 = in-distribution background.
    We treat any pixel with value > 0 and < 255 as OOD.
    (255 is typically the ignore/void label.)
    """
    mask = np.array(Image.open(mask_path))
    return bool(((mask > 0) & (mask < 255)).any())

# ── Collectors ─────────────────────────────────────────────────────────────────
def collect_fishyscapes(model, device, method):
    """
    FishyScapes: RGB and mask files have IDENTICAL filenames.
    RGB in FISHY_RGB, mask in FISHY_MASKS — just swap the folder.
    """
    results = []
    for rgb_path in sorted(FISHY_RGB.rglob("*.png")):
        mask_path = FISHY_MASKS / rgb_path.name  # same filename, different folder
        if not mask_path.exists():
            print(f"  FISHY no mask: {rgb_path.name}")
            continue
        try:
            is_ood = mask_has_ood(mask_path)
            score = image_score(model, rgb_path, device, method)
            results.append((score, is_ood))
        except Exception as e:
            print(f"  FISHY error {rgb_path.name}: {e}")

    n_ood  = sum(r[1] for r in results)
    n_in   = sum(not r[1] for r in results)
    print(f"  FishyScapes: {len(results)} images  ({n_ood} OOD, {n_in} in-dist)")
    return results


def collect_cityscapes(model, device, method, n=100):
    """
    Cityscapes val: all in-distribution by definition.
    Files named *_leftImg8bit.png inside city subfolders.
    """
    all_imgs = sorted(CITY_VAL.rglob("*_leftImg8bit.png"))
    if not all_imgs:
        print(f"  WARNING: no Cityscapes images found at {CITY_VAL}")
        return []

    results = []
    for p in all_imgs[:n]:
        try:
            results.append((image_score(model, p, device, method), False))
        except Exception as e:
            print(f"  CITY skipping {p.name}: {e}")

    print(f"  Cityscapes val: {len(results)} in-distribution images")
    return results


def collect_roadanomaly(model, device, method):
    """
    RoadAnomaly21: every image contains an anomalous object.
    All images are OOD by dataset definition — no masks needed.
    """
    all_imgs = (sorted(ROAD_RGB.rglob("*.jpg")) +
                sorted(ROAD_RGB.rglob("*.png")))
    results = []
    for p in all_imgs:
        try:
            results.append((image_score(model, p, device, method), True))
        except Exception as e:
            print(f"  ROAD skipping {p.name}: {e}")

    print(f"  RoadAnomaly21: {len(results)} images (all OOD)")
    return results

# ── Threshold sweep ────────────────────────────────────────────────────────────
def tune(results, label="validation"):
    scores = np.array([r[0] for r in results], dtype=float)
    labels = np.array([r[1] for r in results], dtype=bool)

    n_in  = (~labels).sum()
    n_ood = labels.sum()

    print(f"\n── {label} ───────────────────────────────────────────────────────")
    print(f"  {len(results)} images  ({n_in} in-dist, {n_ood} OOD)")

    if n_in == 0 or n_ood == 0:
        print("  ERROR: need both classes to tune. Check data paths.")
        return None

    # Show score distributions — useful for paper
    print(f"  Score in-dist : "
          f"mean={scores[~labels].mean():.3f}  "
          f"std={scores[~labels].std():.3f}  "
          f"range=[{scores[~labels].min():.3f}, {scores[~labels].max():.3f}]")
    print(f"  Score OOD     : "
          f"mean={scores[labels].mean():.3f}  "
          f"std={scores[labels].std():.3f}  "
          f"range=[{scores[labels].min():.3f}, {scores[labels].max():.3f}]")

    # Sweep 500 candidate thresholds
    thresholds = np.linspace(scores.min(), scores.max(), 500)
    best_t, best_bal = None, -1

    for t in thresholds:
        pred_ood = scores >= t
        acc_in   = (~pred_ood[~labels]).mean()
        acc_ood  = pred_ood[labels].mean()
        bal      = (acc_in + acc_ood) / 2
        if bal > best_bal:
            best_bal, best_t = bal, t

    # Report at best threshold
    pred_ood = scores >= best_t
    acc_in   = (~pred_ood[~labels]).mean()
    acc_ood  = pred_ood[labels].mean()

    print(f"\n  Best threshold : {best_t:.4f}")
    print(f"  Acc_InDist     : {acc_in*100:.1f}%")
    print(f"  Acc_OOD        : {acc_ood*100:.1f}%")
    print(f"  Balanced acc   : {best_bal*100:.1f}%")

    try:
        auroc = roc_auc_score(labels.astype(int), scores)
        print(f"  AUROC          : {auroc:.4f}")
    except Exception as e:
        print(f"  AUROC: {e}")

    return best_t

# ── Evaluation with fixed threshold ───────────────────────────────────────────
def evaluate(results, threshold, label="test"):
    scores = np.array([r[0] for r in results], dtype=float)
    labels    = np.array([r[1] for r in results], dtype=bool)
    pred_ood  = scores >= threshold

    n_in  = (~labels).sum()
    n_ood = labels.sum()
    acc_in  = (~pred_ood[~labels]).mean() if n_in  > 0 else float("nan")
    acc_ood = pred_ood[labels].mean()     if n_ood > 0 else float("nan")

    print(f"\n── {label} (threshold={threshold:.4f}) ──────────────────────────")
    print(f"  {len(results)} images  ({n_in} in-dist, {n_ood} OOD)")
    print(f"  Acc_InDist : {acc_in*100:.1f}%")
    print(f"  Acc_OOD    : {acc_ood*100:.1f}%")
    print(f"  Balanced   : {((acc_in + acc_ood)/2)*100:.1f}%")

    try:
        auroc = roc_auc_score(labels.astype(int), scores)
        print(f"  AUROC      : {auroc:.4f}")
    except Exception as e:
        print(f"  AUROC: {e}")

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")
    print(f"Method: {args.method}")
    print(f"Model path: {args.model_path}")

    # Ensure logit-only methods do not instantiate GSVD expert/novice modules.
    os.environ["OOD_METHOD"] = args.method

    # Load model
    model = Model()
    load_weights(model, args.model_path, device)
    model.eval().to(device)

    # ── Validation: FishyScapes + Cityscapes ──────────────────────────────────
    print("\n── Collecting validation data ────────────────────────────────────")
    fishy = collect_fishyscapes(model, device, args.method)
    city  = collect_cityscapes(model, device, args.method, n=N_CITY)
    val_results = fishy + city

    threshold = tune(val_results, label="Validation (FishyScapes + Cityscapes val)")

    # ── Test: RoadAnomaly21 ───────────────────────────────────────────────────
    print("\n── Collecting test data ──────────────────────────────────────────")
    road = collect_roadanomaly(model, device, args.method)

    if threshold is not None and road:
        evaluate(road, threshold, label="RoadAnomaly21 (test)")
        print("\n✓ Submission environment variables:")
        print(f"  OOD_METHOD={args.method}")
        print(f"  OOD_THRESHOLD={threshold:.4f}")
    elif threshold is None:
        print("\nCould not determine threshold — check validation data.")