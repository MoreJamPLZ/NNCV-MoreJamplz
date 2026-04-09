"""
Ablation Study 2: Novice Model Complexity for GSVD-based OOD Detection
=======================================================================
Compares three randomly initialized novice architectures paired with
a frozen DINOv3 expert, using the best aggregation metric from Ablation 1 (median).

Novice models (all randomly initialized, same seed):
  A) Linear    — single 1x1 conv projection (3 → 320) + spatial downsample
  B) Nonlinear — shallow CNN (3 conv+BN+ReLU blocks → 320 channels)
  C) SegFormer — full SegFormer-B5 (from Ablation 1 baseline)

All novice models output shape (1, 320, 32, 32) for 512×512 input,
matching SegFormer hidden_states[2].

Expert model:  DINOv3 ViT-B/16 (pretrained, frozen)
Metric:        median (best from Ablation 1)

Outputs (saved to results/ablation2_novice/):
  - novice_comparison.csv            Per-image scores per novice model
  - novice_summary.csv               Accuracy, AUROC, etc. per novice model
  - histogram_novice_comparison.pdf   Publication-quality histograms
  - roc_curves_novice.pdf             ROC curves
  - config.json                       Full experiment config
"""

import os
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
N_RATIOS = 64
START_IDX = 256
METRIC = "median"         # Best from Ablation 1
INPUT_SIZE = (512, 512)
FEAT_DIM = 320            # Must match SegFormer hidden_states[2] channels
FEAT_SIZE = (32, 32)      # Spatial size at 1/16 of 512
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

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
OUT_DIR = "results/ablation2_novice"


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
# Novice Models
# ============================================================================
class LinearNovice(nn.Module):
    """Minimal novice: single linear projection + downsample.
    No learned spatial structure — just projects RGB to FEAT_DIM channels.
    ~960 parameters.
    """
    def __init__(self, in_channels=3, feat_dim=FEAT_DIM):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, feat_dim, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.proj(x)  # (1, 320, 512, 512)
        out = F.interpolate(out, size=FEAT_SIZE, mode="bilinear", align_corners=False)
        return out  # (1, 320, 32, 32)


class NonlinearNovice(nn.Module):
    """Shallow CNN novice: 4 conv blocks with BN + ReLU, progressive downsampling.
    Has spatial structure and nonlinearity, but much simpler than SegFormer.
    ~1.2M parameters.
    """
    def __init__(self, in_channels=3, feat_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            # Block 1: 512 → 256
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Block 2: 256 → 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Block 3: 128 → 64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Block 4: 64 → 32
            nn.Conv2d(256, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)  # (1, 320, 32, 32)


class SegFormerNovice(nn.Module):
    """Full SegFormer-B5 novice (randomly initialized).
    Complex transformer-based encoder. ~84M parameters.
    Returns hidden_states[2] to match Ablation 1.
    """
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
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

    def forward(self, x):
        out = self.segformer(pixel_values=x, output_hidden_states=True)
        return out.hidden_states[2]  # (1, 320, 32, 32)


# ============================================================================
# Expert (DINOv3) + feature extraction
# ============================================================================
class ExpertModel(nn.Module):
    """Frozen DINOv3 — shared across all novice comparisons."""
    def __init__(self):
        super().__init__()
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

    def forward(self, x, target_size):
        """Returns expert features aligned to target spatial size."""
        out = self.dino.forward_features(x)
        tokens = out[:, 5:, :]  # skip register/CLS
        N = tokens.shape[1]
        h = w = int(N ** 0.5)
        grid = tokens.permute(0, 2, 1).reshape(1, 768, h, w)
        aligned = F.interpolate(grid, size=target_size, mode="bilinear", align_corners=False)
        return aligned  # (1, 768, H, W)


# ============================================================================
# Evaluation
# ============================================================================
def find_optimal_threshold(id_scores, ood_scores):
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
    sorted_s = np.sort(np.unique(scores))
    candidates = (sorted_s[:-1] + sorted_s[1:]) / 2
    best_acc, best_t, best_cm = 0.0, 0.0, (0, 0, 0, 0)
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


def compute_gsvd_score(A_feat, B_feat, start_idx, n_ratios, metric):
    """Extract features, run GSVD, return aggregated score."""
    A = A_feat.squeeze(0).flatten(1).T  # (HW, C_A)
    B = B_feat.squeeze(0).flatten(1).T  # (HW, 768)
    A_norm = F.normalize(A, dim=1).T.cpu().numpy()
    B_norm = F.normalize(B, dim=1).T.cpu().numpy()
    _, _, _, C, S = gsvd0(A_norm, B_norm)
    gen_sv = C / S
    ratios = gen_sv[start_idx : start_idx + n_ratios]
    finite = ratios[np.isfinite(ratios)]
    if len(finite) == 0:
        return None
    if metric == "mean":
        return float(finite.mean())
    elif metric == "median":
        return float(np.median(finite))
    elif metric == "max":
        return float(finite.max())


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    novice_configs = {
        "Linear": {
            "class": LinearNovice,
            "color": "#55A868",
            "marker": "^",
            "params_desc": "1×1 conv (3→320) + bilinear downsample",
        },
        "Nonlinear CNN": {
            "class": NonlinearNovice,
            "color": "#C44E52",
            "marker": "s",
            "params_desc": "4-block CNN (3→64→128→256→320, stride-2)",
        },
        "SegFormer-B5": {
            "class": SegFormerNovice,
            "color": "#4C72B0",
            "marker": "o",
            "params_desc": "Full SegFormer-B5 encoder+decoder",
        },
    }

    # Save config
    config = {
        "seed": SEED,
        "n_ratios": N_RATIOS,
        "start_idx": START_IDX,
        "metric": METRIC,
        "input_size": list(INPUT_SIZE),
        "device": DEVICE,
        "expert_model": "vit_base_patch16_dinov3.lvd1689m (timm)",
        "novice_models": {k: v["params_desc"] for k, v in novice_configs.items()},
        "feature_output_shape": f"(1, {FEAT_DIM}, {FEAT_SIZE[0]}, {FEAT_SIZE[1]})",
        "id_folders": ID_FOLDERS,
        "ood_folders": OOD_FOLDERS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Device: {DEVICE}")

    # Build expert (shared)
    expert = ExpertModel().to(DEVICE)
    expert.eval()

    preprocess = Compose([
        ToImage(),
        Resize(size=INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # Collect paths once
    all_paths = {}
    for label, folders in [("ID", ID_FOLDERS), ("OOD", OOD_FOLDERS)]:
        all_paths[label] = collect_image_paths(folders)
        print(f"  {label}: {len(all_paths[label])} images")

    # ---- Run each novice ----
    all_records = {}  # novice_name -> list of {label, filename, score}

    for novice_name, ncfg in novice_configs.items():
        print(f"\n{'=' * 60}")
        print(f"  NOVICE: {novice_name}")
        print(f"  {ncfg['params_desc']}")
        print(f"{'=' * 60}")

        # Initialize novice with same seed
        set_seed(SEED)
        novice = ncfg["class"]().to(DEVICE)
        novice.eval()

        # Count parameters
        n_params = sum(p.numel() for p in novice.parameters())
        print(f"  Parameters: {n_params:,}")

        records = []
        for label in ["ID", "OOD"]:
            paths = all_paths[label]
            print(f"\n  {label}: {len(paths)} images")
            for i, path in enumerate(paths):
                try:
                    img = Image.open(path).convert("RGB")
                    x = preprocess(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        A_feat = novice(x)  # (1, 320, 32, 32)
                        B_feat = expert(x, target_size=(A_feat.shape[2], A_feat.shape[3]))
                    score = compute_gsvd_score(A_feat, B_feat, START_IDX, N_RATIOS, METRIC)
                    if score is None:
                        print(f"    [{i+1:3d}/{len(paths)}] SKIP: {os.path.basename(path)}")
                        continue
                    records.append({
                        "novice": novice_name,
                        "label": label,
                        "filename": os.path.basename(path),
                        "score": score,
                    })
                    if (i + 1) % 50 == 0 or i == len(paths) - 1:
                        print(f"    [{i+1:3d}/{len(paths)}] score={score:7.3f}  {os.path.basename(path)}")
                except Exception as e:
                    print(f"    [{i+1:3d}/{len(paths)}] FAILED: {os.path.basename(path)} — {e}")

        all_records[novice_name] = records
        # Free GPU memory
        del novice
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Save per-image CSV ----
    csv_path = os.path.join(OUT_DIR, "novice_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["novice", "label", "filename", "score"])
        writer.writeheader()
        for records in all_records.values():
            writer.writerows(records)
    print(f"\nSaved per-image results → {csv_path}")

    # ---- Evaluate ----
    summary_rows = []
    for novice_name, records in all_records.items():
        id_vals = np.array([r["score"] for r in records if r["label"] == "ID"])
        ood_vals = np.array([r["score"] for r in records if r["label"] == "OOD"])

        thresh, acc, (tp, fp, fn, tn) = find_optimal_threshold(id_vals, ood_vals)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        labels = np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))])
        scores = np.concatenate([id_vals, ood_vals])
        auroc = roc_auc_score(labels, scores)

        n_params = sum(p.numel() for p in novice_configs[novice_name]["class"]().parameters())

        summary_rows.append({
            "novice": novice_name,
            "params": n_params,
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

    # Print summary
    print(f"\n{'=' * 90}")
    print(f"  NOVICE MODEL COMPARISON (metric={METRIC})")
    print(f"{'=' * 90}")
    print(f"  {'Novice':<16} {'Params':>10} {'Acc':>7} {'AUROC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"  {'-' * 75}")
    for r in summary_rows:
        print(f"  {r['novice']:<16} {r['params']:>10,} {r['accuracy']:>6.1%} {r['auroc']:>6.1%} {r['precision']:>6.1%} {r['recall']:>6.1%} {r['f1']:>6.1%}")

    # Save summary CSV
    summary_csv = os.path.join(OUT_DIR, "novice_summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"\nSaved summary → {summary_csv}")

    # ---- Figure 1: Histograms ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    for ax, (novice_name, ncfg) in zip(axes, novice_configs.items()):
        records = all_records[novice_name]
        id_vals = np.array([r["score"] for r in records if r["label"] == "ID"])
        ood_vals = np.array([r["score"] for r in records if r["label"] == "OOD"])
        row = next(r for r in summary_rows if r["novice"] == novice_name)

        all_vals = np.concatenate([id_vals, ood_vals])
        bins = np.linspace(all_vals.min() * 0.95, all_vals.max() * 1.05, 25)

        ax.hist(id_vals, bins=bins, alpha=0.65, label="In-distribution", color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.hist(ood_vals, bins=bins, alpha=0.65, label="Out-of-distribution", color="#DD4444", edgecolor="white", linewidth=0.5)
        ax.axvline(row["threshold"], color="black", linestyle="--", linewidth=1.2,
                   label=f"Threshold (acc={row['accuracy']:.1%})")
        ax.set_xlabel(f"Median GSVD ratio", fontsize=11)
        ax.set_ylabel("Number of images", fontsize=11)
        ax.set_title(f"{novice_name}\nAUROC={row['auroc']:.3f}  |  {row['params']:,} params",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Ablation 2: Novice Model Complexity (metric={METRIC}, seed={SEED})",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, "histogram_novice_comparison.pdf")
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    fig.savefig(hist_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histograms → {hist_path}")

    # ---- Figure 2: ROC curves ----
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for novice_name, ncfg in novice_configs.items():
        records = all_records[novice_name]
        id_vals = np.array([r["score"] for r in records if r["label"] == "ID"])
        ood_vals = np.array([r["score"] for r in records if r["label"] == "OOD"])
        labels = np.concatenate([np.zeros(len(id_vals)), np.ones(len(ood_vals))])
        scores = np.concatenate([id_vals, ood_vals])
        fpr, tpr, _ = roc_curve(labels, scores)
        auroc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=ncfg["color"], linewidth=2,
                label=f"{novice_name} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves: Novice Model Complexity", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    roc_path = os.path.join(OUT_DIR, "roc_curves_novice.pdf")
    fig.savefig(roc_path, dpi=300, bbox_inches="tight")
    fig.savefig(roc_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curves → {roc_path}")

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()