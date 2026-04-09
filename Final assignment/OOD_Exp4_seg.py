"""
Ablation Study 4: Expert Model Comparison — SegFormer (pretrained) as Expert
=============================================================================
Same setup as Ablations 2+3 but replaces DINOv3 with pretrained SegFormer-B5
as the expert model. Tests whether a task-specific expert works for GSVD OOD.

Expert model:  SegFormer-B5 (pretrained on Cityscapes, frozen)
Metric:        median (from Ablation 1)
Novice models: Nonlinear CNN, SegFormer-B5 (random init)

Outputs (saved to results/ablation4_segformer_expert/):
  - comparison.csv / summary.csv
  - nratios_sweep.csv
  - histogram + ROC figures
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

from transformers import SegformerForSemanticSegmentation, SegformerConfig
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode,
)

# ============================================================================
# Configuration
# ============================================================================
SEED = 1
START_IDX = 100           # Scaled from 256/320 ≈ 0.8 → 0.8 * 128 = 102, rounded to 100
MAX_RATIOS = 28           # Only 128 - 100 = 28 values available (vs 64 with DINOv3)
N_RATIOS_LIST = [2, 5, 8, 10, 15, 20, 25, 28]
METRIC = "median"
INPUT_SIZE = (512, 512)
NOVICE_FEAT_DIM = 128     # Novice output dim (~320/2.4, preserving 768:320 ratio from DINOv3 setup)
FEAT_SIZE = (32, 32)
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

BASE = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz"
SEGFORMER_WEIGHTS = f"{BASE}/segformer.pt"  # pretrained SegFormer-B5 checkpoint

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
OUT_DIR = "results/ablation4_segformer_expert"


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
def gsvd0(A, B):
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
# Expert: Pretrained SegFormer-B5
# ============================================================================
class SegFormerExpert(nn.Module):
    """Pretrained SegFormer-B5 as expert. Returns hidden_states[2] (320-dim)."""
    def __init__(self, weights_path):
        super().__init__()
        config = SegformerConfig(
            num_channels=3, num_labels=19,
            num_encoder_blocks=4, depths=[3, 6, 40, 3],
            sr_ratios=[8, 4, 2, 1], hidden_sizes=[64, 128, 320, 512],
            num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            hidden_act="gelu", hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
            decoder_hidden_size=768, semantic_loss_ignore_index=255,
        )
        self.segformer = SegformerForSemanticSegmentation(config)

        # Load pretrained weights
        print(f"  Loading expert SegFormer weights from: {weights_path}")
        state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

        # Remove non-SegFormer keys if present (e.g. ood_threshold from wrapper)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("ood_")}

        missing, unexpected = self.segformer.load_state_dict(state_dict, strict=False)
        loaded = len(state_dict) - len(unexpected)
        print(f"  Loaded {loaded}/{len(state_dict)} keys, {len(missing)} missing, {len(unexpected)} unexpected")
        if loaded < 100:
            print(f"  WARNING: Very few keys loaded! Check key format.")
            print(f"  First 3 checkpoint keys: {list(state_dict.keys())[:3]}")
            print(f"  First 3 expected keys:   {list(self.segformer.state_dict().keys())[:3]}")

        self.segformer.eval()
        for p in self.segformer.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        """Returns hidden_states[2]: (1, 320, H/16, W/16)."""
        out = self.segformer(pixel_values=x, output_hidden_states=True)
        return out.hidden_states[2]


# ============================================================================
# Novice Models
# ============================================================================
class NonlinearNovice(nn.Module):
    def __init__(self, in_channels=3, feat_dim=NOVICE_FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SegFormerNovice(nn.Module):
    """Random SegFormer-B5 novice. Returns hidden_states[1] (128-dim)
    to maintain expert:novice ratio similar to DINOv3(768):SegFormer(320)."""
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
        feat = out.hidden_states[1]  # (1, 128, H/8, W/8)
        # Downsample to match expert spatial size (32x32)
        feat = F.interpolate(feat, size=FEAT_SIZE, mode="bilinear", align_corners=False)
        return feat  # (1, 128, 32, 32)


# ============================================================================
# Utilities
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


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    novice_configs = {
        "Nonlinear CNN": {"class": NonlinearNovice, "color": "#C44E52", "marker": "s"},
        "SegFormer-B5 (rand)": {"class": SegFormerNovice, "color": "#4C72B0", "marker": "o"},
    }

    config = {
        "seed": SEED,
        "start_idx": START_IDX,
        "max_ratios": MAX_RATIOS,
        "n_ratios_list": N_RATIOS_LIST,
        "metric": METRIC,
        "input_size": list(INPUT_SIZE),
        "device": DEVICE,
        "expert_model": "SegFormer-B5 (pretrained Cityscapes)",
        "expert_weights": SEGFORMER_WEIGHTS,
        "expert_feature_layer": "hidden_states[2] (320-dim)",
        "novice_feature_dim": NOVICE_FEAT_DIM,
        "dim_ratio": f"expert(320):novice({NOVICE_FEAT_DIM}) ≈ {320/NOVICE_FEAT_DIM:.1f}:1",
        "novice_models": list(novice_configs.keys()),
        "id_folders": ID_FOLDERS,
        "ood_folders": OOD_FOLDERS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Device: {DEVICE}")

    # Build expert
    expert = SegFormerExpert(SEGFORMER_WEIGHTS).to(DEVICE)
    expert.eval()

    preprocess = Compose([
        ToImage(),
        Resize(size=INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    all_paths = {}
    for label, folders in [("ID", ID_FOLDERS), ("OOD", OOD_FOLDERS)]:
        all_paths[label] = collect_image_paths(folders)
        print(f"  {label}: {len(all_paths[label])} images")

    # ---- Run each novice, store all 64 ratios ----
    raw_data = {}

    for novice_name, ncfg in novice_configs.items():
        print(f"\n{'=' * 60}")
        print(f"  NOVICE: {novice_name}")
        print(f"{'=' * 60}")

        set_seed(SEED)
        novice = ncfg["class"]().to(DEVICE)
        novice.eval()

        n_params = sum(p.numel() for p in novice.parameters())
        print(f"  Parameters: {n_params:,}")

        entries = []
        for label in ["ID", "OOD"]:
            paths = all_paths[label]
            print(f"\n  {label}: {len(paths)} images")
            for i, path in enumerate(paths):
                try:
                    img = Image.open(path).convert("RGB")
                    x = preprocess(img).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        A_feat = novice(x)       # (1, 128, 32, 32)
                        B_feat = expert(x)        # (1, 320, 32, 32)

                    A = A_feat.squeeze(0).flatten(1).T
                    B = B_feat.squeeze(0).flatten(1).T
                    A_norm = F.normalize(A, dim=1).T.cpu().numpy()  # (novice_dim, 1024)
                    B_norm = F.normalize(B, dim=1).T.cpu().numpy()  # (320, 1024)

                    _, _, _, C, S = gsvd0(A_norm, B_norm)
                    gen_sv = C / S
                    all_ratios = gen_sv[START_IDX : START_IDX + MAX_RATIOS].copy()

                    entries.append({
                        "label": label,
                        "filename": os.path.basename(path),
                        "all_ratios": all_ratios,
                    })

                    if (i + 1) % 100 == 0 or i == len(paths) - 1:
                        finite = all_ratios[np.isfinite(all_ratios)]
                        med = float(np.median(finite)) if len(finite) > 0 else 0.0
                        print(f"    [{i+1:3d}/{len(paths)}] median={med:7.3f}  {os.path.basename(path)}")
                except Exception as e:
                    print(f"    [{i+1:3d}/{len(paths)}] FAILED: {os.path.basename(path)} — {e}")

        raw_data[novice_name] = entries
        del novice
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ---- Save per-image ratios ----
    raw_csv = os.path.join(OUT_DIR, "per_image_ratios.csv")
    with open(raw_csv, "w", newline="") as f:
        header = ["novice", "label", "filename"] + [f"ratio_{j+1}" for j in range(MAX_RATIOS)]
        writer = csv.writer(f)
        writer.writerow(header)
        for novice_name, entries in raw_data.items():
            for e in entries:
                row = [novice_name, e["label"], e["filename"]]
                row += [f"{r:.6f}" for r in e["all_ratios"]]
                writer.writerow(row)
    print(f"\nSaved per-image ratios → {raw_csv}")

    # ---- Sweep n_ratios ----
    sweep_rows = []

    print(f"\n{'=' * 90}")
    print(f"  SWEEP: n_ratios × novice (expert=SegFormer pretrained, metric={METRIC})")
    print(f"{'=' * 90}")
    print(f"  {'Novice':<22} {'n_ratios':>8} {'Acc':>7} {'AUROC':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'Thresh':>10}")
    print(f"  {'-' * 85}")

    for novice_name, entries in raw_data.items():
        for n_ratios in N_RATIOS_LIST:
            id_scores, ood_scores = [], []
            for e in entries:
                subset = e["all_ratios"][:n_ratios]
                finite = subset[np.isfinite(subset)]
                if len(finite) == 0:
                    continue
                score = float(np.median(finite))
                if e["label"] == "ID":
                    id_scores.append(score)
                else:
                    ood_scores.append(score)

            id_scores = np.array(id_scores)
            ood_scores = np.array(ood_scores)

            thresh, acc, (tp, fp, fn, tn) = find_optimal_threshold(id_scores, ood_scores)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            labels_arr = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
            scores_arr = np.concatenate([id_scores, ood_scores])
            auroc = roc_auc_score(labels_arr, scores_arr)

            sweep_rows.append({
                "novice": novice_name,
                "n_ratios": n_ratios,
                "threshold": round(thresh, 4),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "auroc": round(auroc, 4),
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            })

            print(f"  {novice_name:<22} {n_ratios:>8} {acc:>6.1%} {auroc:>6.1%} {prec:>6.1%} {rec:>6.1%} {f1:>6.1%} {thresh:>10.4f}")

    # Save sweep
    sweep_csv = os.path.join(OUT_DIR, "nratios_sweep.csv")
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\nSaved sweep → {sweep_csv}")

    # ---- Figure 1: Accuracy + AUROC vs n_ratios ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, y_key, y_label in zip(axes, ["accuracy", "auroc"], ["Accuracy", "AUROC"]):
        for novice_name, ncfg in novice_configs.items():
            rows = [r for r in sweep_rows if r["novice"] == novice_name]
            ns = [r["n_ratios"] for r in rows]
            vals = [r[y_key] for r in rows]
            ax.plot(ns, vals, color=ncfg["color"], marker=ncfg["marker"],
                    linewidth=2, markersize=6, label=novice_name)
            best_idx = np.argmax(vals)
            ax.annotate(f"{vals[best_idx]:.1%}",
                        xy=(ns[best_idx], vals[best_idx]),
                        textcoords="offset points", xytext=(0, 10),
                        fontsize=8, ha="center", fontweight="bold",
                        color=ncfg["color"])

        ax.set_xlabel("Number of GSVD ratios", fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.set_title(f"{y_label} vs. Number of Ratios", fontsize=12, fontweight="bold")
        ax.set_xticks(N_RATIOS_LIST)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)
        ax.tick_params(labelsize=9)

    fig.suptitle(
        f"Expert: SegFormer-B5 (pretrained) — metric={METRIC}, seed={SEED}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plot_path = os.path.join(OUT_DIR, "nratios_sweep.pdf")
    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
    fig.savefig(plot_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sweep plot → {plot_path}")

    # ---- Figure 2: ROC curves at best n_ratios ----
    fig, ax = plt.subplots(figsize=(5.5, 5))

    for novice_name, ncfg in novice_configs.items():
        rows = [r for r in sweep_rows if r["novice"] == novice_name]
        best_row = max(rows, key=lambda r: r["auroc"])
        best_n = best_row["n_ratios"]

        id_scores, ood_scores = [], []
        for e in raw_data[novice_name]:
            subset = e["all_ratios"][:best_n]
            finite = subset[np.isfinite(subset)]
            if len(finite) == 0:
                continue
            score = float(np.median(finite))
            if e["label"] == "ID":
                id_scores.append(score)
            else:
                ood_scores.append(score)

        labels_arr = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        scores_arr = np.concatenate([id_scores, ood_scores])
        fpr, tpr, _ = roc_curve(labels_arr, scores_arr)
        auroc = roc_auc_score(labels_arr, scores_arr)
        ax.plot(fpr, tpr, color=ncfg["color"], linewidth=2,
                label=f"{novice_name} n={best_n} (AUROC={auroc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC: Expert=SegFormer (pretrained)", fontsize=12, fontweight="bold")
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

    # ---- Figure 3: Best histograms ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, (novice_name, ncfg) in zip(axes, novice_configs.items()):
        rows = [r for r in sweep_rows if r["novice"] == novice_name]
        best_row = max(rows, key=lambda r: r["auroc"])
        best_n = best_row["n_ratios"]

        id_scores, ood_scores = [], []
        for e in raw_data[novice_name]:
            subset = e["all_ratios"][:best_n]
            finite = subset[np.isfinite(subset)]
            if len(finite) == 0:
                continue
            score = float(np.median(finite))
            if e["label"] == "ID":
                id_scores.append(score)
            else:
                ood_scores.append(score)

        id_scores = np.array(id_scores)
        ood_scores = np.array(ood_scores)
        all_vals = np.concatenate([id_scores, ood_scores])
        bins = np.linspace(all_vals.min() * 0.95, all_vals.max() * 1.05, 25)

        ax.hist(id_scores, bins=bins, alpha=0.65, label="In-distribution",
                color="#4C72B0", edgecolor="white", linewidth=0.5)
        ax.hist(ood_scores, bins=bins, alpha=0.65, label="Out-of-distribution",
                color="#DD4444", edgecolor="white", linewidth=0.5)
        ax.axvline(best_row["threshold"], color="black", linestyle="--", linewidth=1.2,
                   label=f"Threshold (acc={best_row['accuracy']:.1%})")
        ax.set_xlabel("Median GSVD ratio", fontsize=11)
        ax.set_ylabel("Number of images", fontsize=11)
        ax.set_title(f"{novice_name} (n={best_n})\nAUROC={best_row['auroc']:.3f}",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right")
        ax.tick_params(labelsize=9)

    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, "histogram_best.pdf")
    fig.savefig(hist_path, dpi=300, bbox_inches="tight")
    fig.savefig(hist_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved histograms → {hist_path}")

    print(f"\n{'=' * 60}")
    print(f"  All outputs saved to: {OUT_DIR}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()