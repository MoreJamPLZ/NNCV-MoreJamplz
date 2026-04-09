"""
Ablation Study 1: Aggregation Metric Comparison for GSVD-based OOD Detection
=============================================================================
Compares mean, median, and max aggregation of generalized singular value ratios
for distinguishing in-distribution (Cityscapes) from out-of-distribution images.

Expert model:  DINOv3 ViT-B/16 (pretrained, frozen)
Novice model:  SegFormer-B5 (randomly initialized, fixed seed)

Outputs (saved to results/ablation1_metric/):
  - metric_comparison.csv          Per-image scores for all metrics
  - metric_summary.csv             Accuracy, precision, recall, F1, AUROC per metric
  - histogram_metric_comparison.pdf  Publication-quality histogram figure
  - roc_curves.pdf                  ROC curves for all metrics
  - config.json                     Full experiment configuration for reproducibility
"""

import os
import sys
import json
import glob
import csv
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

import timm
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode,
)

# ============================================================================
# Configuration
# ============================================================================
SEED = 1
N_RATIOS = 64            # Number of GSVD ratios to use
START_IDX = 256           # Starting index in the generalized singular value spectrum
INPUT_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# -- Paths (EDIT THESE) --
BASE = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz"
ID_FOLDERS = [
    f"{BASE}/data/cityscapes/leftImg8bit/val/tubingen",
    f"{BASE}/data/cityscapes/leftImg8bit/val/ulm",
    f"{BASE}/data/cityscapes/leftImg8bit/val/weimar",
    f"{BASE}/data/cityscapes/leftImg8bit/val/zurich",
]
OOD_FOLDERS = [
    f"{BASE}/fishyscapes_rgb_100",
    f"{BASE}/dataset_AnomalyTrack/images",
]

OUT_DIR = "results/ablation1_metric"

# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# GSVD
# ============================================================================
def gsvd0(A: np.ndarray, B: np.ndarray):
    """Reduced GSVD (Paige & Saunders, 1981)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    M = np.vstack([A, B])
    Q, R = np.linalg.qr(M, mode="reduced")
    m1 = A.shape[0]
    Q1, Q2 = Q[:m1, :], Q[m1:, :]
    U, c, Wt = np.linalg.svd(Q1, full_matrices=False)
    W = Wt.T
    s = np.sqrt(np.maximum(1.0 - c ** 2, 0.0))
    Q2W = Q2 @ W
    V = np.zeros_like(Q2W)
    for i in range(Q2W.shape[1]):
        nrm = np.linalg.norm(Q2W[:, i])
        if nrm > 1e-14:
            V[:, i] = Q2W[:, i] / nrm
    X = R.T @ W
    return U, V, X, c, s

# ============================================================================
# Model
# ============================================================================
class GSVDModel(nn.Module):
    """DINOv3 (expert, frozen) + randomly initialized SegFormer (novice)."""

    def __init__(self, in_channels=3, n_classes=19, seed=42):
        super().__init__()
        # Novice: randomly initialized SegFormer-B5
        set_seed(seed)
        config = SegformerConfig(
            num_channels=in_channels, num_labels=n_classes,
            num_encoder_blocks=4, depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1], hidden_sizes=[64, 128, 320, 512],
            num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu", hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
            decoder_hidden_size=768, semantic_loss_ignore_index=255,
        )
        self.segformer = SegformerForSemanticSegmentation(config)

        # Expert: pretrained DINOv3
        self.dino = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=False,
            num_classes=0,
        )
        dino_weights = f"{BASE}/dinov3_vitb16_timm.pth"
        state_dict = torch.load(dino_weights, map_location="cpu", weights_only=True)
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.dino.load_state_dict(state_dict, strict=True)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

    def extract_features(self, x):
        """Returns aligned feature matrices A (novice) and B (expert)."""
        seg_out = self.segformer(pixel_values=x, output_hidden_states=True)
        logits = seg_out.logits
        A_spatial = seg_out.hidden_states[2]  # (1, 320, H, W)

        dino_out = self.dino.forward_features(x)  # returns tensor directly
        B_tokens = dino_out[:, 5:, :]  # skip register/CLS tokens
        N = B_tokens.shape[1]
        h = w = int(N ** 0.5)
        B_grid = B_tokens.permute(0, 2, 1).reshape(1, 768, h, w)
        B_aligned = F.interpolate(
            B_grid, size=(A_spatial.shape[2], A_spatial.shape[3]),
            mode="bilinear", align_corners=False,
        )

        A = A_spatial.squeeze(0).flatten(1).T  # (HW, 320)
        B = B_aligned.squeeze(0).flatten(1).T  # (HW, 768)

        A_norm = F.normalize(A, dim=1).T.cpu().numpy()  # (320, HW)
        B_norm = F.normalize(B, dim=1).T.cpu().numpy()  # (768, HW)

        return logits, A_norm, B_norm

    def compute_ratios(self, A_norm, B_norm, start_idx=256, n_ratios=10):
        """Compute GSVD and return the selected generalized singular value ratios."""
        _, _, _, C, S = gsvd0(A_norm, B_norm)
        gen_sv = C / S
        return gen_sv[start_idx : start_idx + n_ratios]

# ============================================================================
# Evaluation utilities
# ============================================================================
def find_optimal_threshold(id_scores, ood_scores):
    """Sweep thresholds to maximize accuracy."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    sorted_s = np.sort(np.unique(scores))
    candidates = (sorted_s[:-1] + sorted_s[1:]) / 2

    best_acc, best_t = 0.0, 0.0
    best_cm = (0, 0, 0, 0)
    for t in candidates:
        preds = (scores > t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
            tp = int(((preds == 1) & (labels == 1)).sum())
            fp = int(((preds == 1) & (labels == 0)).sum())
            fn = int(((preds == 0) & (labels == 1)).sum())
            tn = int(((preds == 0) & (labels == 0)).sum())
            best_cm = (tp, fp, fn, tn)
    return best_t, best_acc, best_cm


def collect_image_paths(folders):
    paths = []
    for folder in folders:
        paths += sorted(glob.glob(os.path.join(folder, "*.png")))
        paths += sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return paths


# ============================================================================
# Main experiment
# ============================================================================
def main():
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save config
    config = {
        "seed": SEED,
        "n_ratios": N_RATIOS,
        "start_idx": START_IDX,
        "input_size": list(INPUT_SIZE),
        "device": DEVICE,
        "expert_model": "vit_base_patch16_dinov3.lvd1689m (timm)",
        "novice_model": "SegFormer-B5 (random init)",
        "novice_config": "depths=[3,6,40,3], hidden_sizes=[64,128,320,512]",
        "feature_layer": "hidden_states[2] (320-dim)",
        "id_folders": ID_FOLDERS,
        "ood_folders": OOD_FOLDERS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Build model
    print(f"Device: {DEVICE}")
    model = GSVDModel(seed=SEED).to(DEVICE)
    model.eval()

    preprocess = Compose([
        ToImage(),
        Resize(size=INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # ---- Inference ----
    records = []  # list of dicts: label, filename, mean, median, max
    for label, folders in [("ID", ID_FOLDERS), ("OOD", OOD_FOLDERS)]:
        paths = collect_image_paths(folders)
        print(f"\n{'=' * 60}\n  {label}: {len(paths)} images\n{'=' * 60}")
        for i, path in enumerate(paths):
            try:
                img = Image.open(path).convert("RGB")
                x = preprocess(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    _, A_norm, B_norm = model.extract_features(x)
                ratios = model.compute_ratios(A_norm, B_norm, START_IDX, N_RATIOS)
                finite = ratios[np.isfinite(ratios)]
                if len(finite) == 0:
                    print(f"  [{i+1:3d}/{len(paths)}] SKIP (no finite ratios): {os.path.basename(path)}")
                    continue
                rec = {
                    "label": label,
                    "filename": os.path.basename(path),
                    "mean": float(finite.mean()),
                    "median": float(np.median(finite)),
                    "max": float(finite.max()),
                }
                records.append(rec)
                print(f"  [{i+1:3d}/{len(paths)}] mean={rec['mean']:6.3f}  median={rec['median']:6.3f}  max={rec['max']:7.3f}  {rec['filename']}")
            except Exception as e:
                print(f"  [{i+1:3d}/{len(paths)}] FAILED: {os.path.basename(path)} — {e}")

    # ---- Save per-image CSV ----
    csv_path = os.path.join(OUT_DIR, "metric_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "filename", "mean", "median", "max"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved per-image results → {csv_path}")

    # ---- Evaluate each metric ----
    metrics = ["mean", "median", "max"]
    summary_rows = []

    id_records = [r for r in records if r["label"] == "ID"]
    ood_records = [r for r in records if r["label"] == "OOD"]

    for metric in metrics:
        id_vals = np.array([r[metric] for r in id_records])
        ood_vals = np.array([r[metric] for r in ood_records])

        thresh, acc, (tp, fp, fn, tn) = find_optimal_threshold(id_vals, ood_vals)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        labels = np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))])
        scores = np.concatenate([id_vals, ood_vals])
        auroc = roc_auc_score(labels, scores)

        summary_rows.append({
            "metric": metric,
            "threshold": round(thresh, 4),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "auroc": round(auroc, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "id_mean": round(id_vals.mean(), 4),
            "id_std": round(id_vals.std(), 4),
            "ood_mean": round(ood_vals.mean(), 4),
            "ood_std": round(ood_vals.std(), 4),
        })

    # Print summary table
    print(f"\n{'=' * 80}")
    print("  METRIC COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Metric':<8} {'Acc':>7} {'AUROC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Thresh':>8}  {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}")
    print(f"  {'-' * 75}")
    for r in summary_rows:
        print(f"  {r['metric']:<8} {r['accuracy']:>6.1%} {r['auroc']:>6.1%} {r['precision']:>6.1%} {r['recall']:>6.1%} {r['f1']:>6.1%} {r['threshold']:>8.4f}  {r['tp']:>3} {r['fp']:>3} {r['fn']:>3} {r['tn']:>3}")

    # Save summary CSV
    summary_csv = os.path.join(OUT_DIR, "metric_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary → {summary_csv}")

    # ---- Figure 1: Histograms (publication quality) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    titles = {"mean": "Mean", "median": "Median", "max": "Max"}

    for ax, metric in zip(axes, metrics):
        id_vals = np.array([r[metric] for r in id_records])
        ood_vals = np.array([r[metric] for r in ood_records])
        row = next(r for r in summary_rows if r["metric"] == metric)

        all_vals = np.concatenate([id_vals, ood_vals])
        bins = np.linspace(all_vals.min() * 0.95, all_vals.max() * 1.05, 25)

        ax.hist(id_vals, bins=bins, alpha=0.65, label="In-distribution", color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.hist(ood_vals, bins=bins, alpha=0.65, label="Out-of-distribution", color="#DD4444", edgecolor="white", linewidth=0.5)
        ax.axvline(row["threshold"], color="black", linestyle="--", linewidth=1.2,
                   label=f"Threshold (acc={row['accuracy']:.1%})")
        ax.set_xlabel(f"{titles[metric]} GSVD ratio", fontsize=11)
        ax.set_ylabel("Number of images", fontsize=11)
        ax.set_title(f"{titles[metric]}  |  AUROC={row['auroc']:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Ablation 1: Aggregation Metric Comparison (n_ratios={N_RATIOS}, seed={SEED})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, "histogram_metric_comparison.pdf")
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    fig.savefig(hist_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histogram → {hist_path}")

    # ---- Figure 2: ROC curves ----
    fig, ax = plt.subplots(figsize=(5.5, 5))
    colors = {"mean": "#4C72B0", "median": "#55A868", "max": "#DD4444"}

    for metric in metrics:
        id_vals = np.array([r[metric] for r in id_records])
        ood_vals = np.array([r[metric] for r in ood_records])
        labels = np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))])
        scores = np.concatenate([id_vals, ood_vals])
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=colors[metric], linewidth=2, label=f"{metric.capitalize()} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Aggregation Metric Comparison", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    roc_path = os.path.join(OUT_DIR, "roc_curves.pdf")
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    fig.savefig(roc_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curves → {roc_path}")

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()