"""
Ablation 5: Domain-Shift Control — does GSVD detect anomalies or domain shift?
==============================================================================
Updated version with:
  1. Density-normalized histograms (comparable across unequal group sizes)
  2. DS_real (BDD100K) included in the ROC panel
  3. OOD split by dataset (Fishyscapes vs AnomalyTrack) for diagnostic AUROCs
  4. Softened "bright" augmentation (less 255-clipping, less fake edge content)
  5. NEW: GSVD score vs mean image luminance scatter — diagnoses whether the
     score reduces to a simple brightness/contrast detector

Groups:
  ID:       clean Cityscapes val
  DS:       synthetically domain-shifted Cityscapes (5 augmentations)
  DS_real:  real domain shift (BDD100K) — no anomalies
  OOD:      Fishyscapes + AnomalyTrack (pooled headline, split diagnostic)

Outputs (results/ablation5_domain_shift/):
  - scores.csv                       per-image scores, incl. subgroup + luminance
  - summary.json                     pairwise AUROCs + per-OOD-dataset breakdown
  - three_way_histogram.pdf          density-normalized, all groups
  - roc_pairwise.pdf                 5 ROC curves incl. DS_real
  - score_vs_augmentation.pdf        per-aug means + DS_real reference line
  - score_vs_luminance.pdf           NEW: diagnostic scatter
  - config.json
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
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score, roc_curve

import timm
from torchvision.transforms.v2 import (
    Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode,
)

# ============================================================================
# Configuration
# ============================================================================
SEED = 1
N_RATIOS = 10
START_IDX = 256
METRIC = "median"
INPUT_SIZE = (512, 512)
FEAT_DIM = 320
FEAT_SIZE = (32, 32)
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
DS_REAL_FOLDERS = [
    f"{BASE}/bdd100k_500",
]

OUT_DIR = "results/ablation5_domain_shift"


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
# Domain-shift augmentations
# (bright softened: * 1.3 + 15 instead of * 1.6 + 30, reduces 255-clipping)
# ============================================================================
class DomainShiftAugmentations:
    @staticmethod
    def dark(img):
        arr = np.asarray(img, dtype=np.float32)
        arr = arr * 0.35
        arr[..., 2] = np.clip(arr[..., 2] * 1.15, 0, 255)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @staticmethod
    def bright(img):
        """Simulate overexposure. Softened from *1.6+30 to *1.3+15 to reduce
        artificial 255-clipped edges that the expert was responding to."""
        arr = np.asarray(img, dtype=np.float32)
        arr = np.clip(arr * 1.3 + 15, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def blur(img):
        return img.filter(ImageFilter.GaussianBlur(radius=4))

    @staticmethod
    def color_cast(img):
        arr = np.asarray(img, dtype=np.float32)
        arr[..., 0] = np.clip(arr[..., 0] * 1.25, 0, 255)
        arr[..., 2] = np.clip(arr[..., 2] * 0.75, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))

    @staticmethod
    def fog(img):
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        arr = np.asarray(img, dtype=np.float32)
        gray = 180.0
        arr = arr * 0.55 + gray * 0.45
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    @classmethod
    def all_augs(cls):
        return {
            "dark":       cls.dark,
            "bright":     cls.bright,
            "blur":       cls.blur,
            "color_cast": cls.color_cast,
            "fog":        cls.fog,
        }


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
# Models
# ============================================================================
class NonlinearNovice(nn.Module):
    def __init__(self, in_channels=3, feat_dim=FEAT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, feat_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ExpertModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dino = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0,
        )
        state_dict = torch.load(
            f"{BASE}/dinov3_vitb16_timm.pth", map_location="cpu", weights_only=True
        )
        if "model" in state_dict:
            state_dict = state_dict["model"]
        self.dino.load_state_dict(state_dict, strict=True)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

    def forward(self, x, target_size):
        out = self.dino.forward_features(x)
        tokens = out[:, 5:, :]
        N = tokens.shape[1]
        h = w = int(N ** 0.5)
        grid = tokens.permute(0, 2, 1).reshape(1, 768, h, w)
        return F.interpolate(grid, size=target_size, mode="bilinear", align_corners=False)


# ============================================================================
# Helpers
# ============================================================================
def collect_image_paths(folders):
    paths = []
    for folder in folders:
        paths += sorted(glob.glob(os.path.join(folder, "*.png")))
        paths += sorted(glob.glob(os.path.join(folder, "*.jpg")))
    return paths


def image_luminance(pil_img):
    """Mean pixel luminance (0-255) computed on the raw PIL image."""
    return float(np.asarray(pil_img.convert("L"), dtype=np.float32).mean())


def ood_subgroup(path):
    p = path.lower()
    if "fishyscapes" in p:
        return "fishyscapes"
    if "anomalytrack" in p or "anomaly_track" in p:
        return "anomalytrack"
    return "unknown"


def compute_score(pil_img, novice, expert, preprocess):
    x = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        A_feat = novice(x)
        B_feat = expert(x, target_size=(A_feat.shape[2], A_feat.shape[3]))
    A = A_feat.squeeze(0).flatten(1).T
    B = B_feat.squeeze(0).flatten(1).T
    A_norm = F.normalize(A, dim=1).T.cpu().numpy()
    B_norm = F.normalize(B, dim=1).T.cpu().numpy()
    _, _, _, C, S = gsvd0(A_norm, B_norm)
    gen_sv = C / S
    ratios = gen_sv[START_IDX : START_IDX + N_RATIOS]
    finite = ratios[np.isfinite(ratios)]
    if len(finite) == 0:
        return None
    return float(np.median(finite))


# ============================================================================
# Main
# ============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    config = {
        "seed": SEED,
        "n_ratios": N_RATIOS,
        "start_idx": START_IDX,
        "metric": METRIC,
        "input_size": list(INPUT_SIZE),
        "device": DEVICE,
        "expert": "DINOv3 ViT-B/16 frozen",
        "novice": "Nonlinear CNN, random-init seed=1",
        "augmentations": list(DomainShiftAugmentations.all_augs().keys()),
        "augmentation_notes": {
            "bright": "softened from *1.6+30 to *1.3+15 to reduce 255-clipping",
        },
        "id_folders": ID_FOLDERS,
        "ood_folders": OOD_FOLDERS,
        "ds_real_folders": DS_REAL_FOLDERS,
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Device: {DEVICE}")

    set_seed(SEED)
    novice = NonlinearNovice().to(DEVICE).eval()
    expert = ExpertModel().to(DEVICE).eval()

    preprocess = Compose([
        ToImage(),
        Resize(size=INPUT_SIZE, interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    id_paths = collect_image_paths(ID_FOLDERS)
    ood_paths = collect_image_paths(OOD_FOLDERS)
    ds_real_paths = collect_image_paths(DS_REAL_FOLDERS) if DS_REAL_FOLDERS else []
    augs = DomainShiftAugmentations.all_augs()

    print(f"  ID:       {len(id_paths)} images")
    print(f"  OOD:      {len(ood_paths)} images")
    print(f"  DS-real:  {len(ds_real_paths)} images")
    print(f"  DS-synth: {len(id_paths)} × {len(augs)} = {len(id_paths)*len(augs)} scores")

    records = []

    # --- ID (clean) ---
    print(f"\n{'='*60}\n  ID (clean Cityscapes)\n{'='*60}")
    for i, path in enumerate(id_paths):
        try:
            img = Image.open(path).convert("RGB")
            lum = image_luminance(img)
            score = compute_score(img, novice, expert, preprocess)
            if score is None: continue
            records.append({
                "group": "ID", "subgroup": "none",
                "filename": os.path.basename(path),
                "score": score, "luminance": lum,
            })
            if (i+1) % 50 == 0: print(f"  [{i+1}/{len(id_paths)}] score={score:.3f} lum={lum:.1f}")
        except Exception as e:
            print(f"  FAIL {os.path.basename(path)}: {e}")

    # --- DS synthetic ---
    for aug_name, aug_fn in augs.items():
        print(f"\n{'='*60}\n  DS (aug={aug_name})\n{'='*60}")
        for i, path in enumerate(id_paths):
            try:
                img = Image.open(path).convert("RGB")
                img_aug = aug_fn(img)
                lum = image_luminance(img_aug)
                score = compute_score(img_aug, novice, expert, preprocess)
                if score is None: continue
                records.append({
                    "group": "DS", "subgroup": aug_name,
                    "filename": os.path.basename(path),
                    "score": score, "luminance": lum,
                })
                if (i+1) % 50 == 0: print(f"  [{i+1}/{len(id_paths)}] score={score:.3f} lum={lum:.1f}")
            except Exception as e:
                print(f"  FAIL {os.path.basename(path)} aug={aug_name}: {e}")

    # --- DS_real ---
    if ds_real_paths:
        print(f"\n{'='*60}\n  DS_real (BDD100K)\n{'='*60}")
        for i, path in enumerate(ds_real_paths):
            try:
                img = Image.open(path).convert("RGB")
                lum = image_luminance(img)
                score = compute_score(img, novice, expert, preprocess)
                if score is None: continue
                records.append({
                    "group": "DS_real", "subgroup": "bdd100k",
                    "filename": os.path.basename(path),
                    "score": score, "luminance": lum,
                })
                if (i+1) % 50 == 0: print(f"  [{i+1}/{len(ds_real_paths)}] score={score:.3f} lum={lum:.1f}")
            except Exception as e:
                print(f"  FAIL {os.path.basename(path)}: {e}")

    # --- OOD (tag by dataset) ---
    print(f"\n{'='*60}\n  OOD (tagged by dataset)\n{'='*60}")
    for i, path in enumerate(ood_paths):
        try:
            img = Image.open(path).convert("RGB")
            lum = image_luminance(img)
            score = compute_score(img, novice, expert, preprocess)
            if score is None: continue
            sg = ood_subgroup(path)
            records.append({
                "group": "OOD", "subgroup": sg,
                "filename": os.path.basename(path),
                "score": score, "luminance": lum,
            })
            if (i+1) % 50 == 0: print(f"  [{i+1}/{len(ood_paths)}] score={score:.3f} lum={lum:.1f} ({sg})")
        except Exception as e:
            print(f"  FAIL {os.path.basename(path)}: {e}")

    # ---- Save CSV ----
    csv_path = os.path.join(OUT_DIR, "scores.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "subgroup", "filename", "score", "luminance"])
        writer.writeheader()
        writer.writerows(records)
    print(f"\nSaved per-image scores → {csv_path}")

    # ---- Build group arrays ----
    groups = {}
    for r in records:
        groups.setdefault(r["group"], []).append(r["score"])
    groups = {k: np.array(v) for k, v in groups.items()}

    ood_by_dataset = {}
    for r in records:
        if r["group"] == "OOD":
            ood_by_dataset.setdefault(r["subgroup"], []).append(r["score"])
    ood_by_dataset = {k: np.array(v) for k, v in ood_by_dataset.items()}

    def auroc(pos, neg):
        y = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
        s = np.concatenate([neg, pos])
        return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")

    summary = {
        "n_per_group": {k: int(len(v)) for k, v in groups.items()},
        "mean_per_group": {k: float(v.mean()) for k, v in groups.items()},
        "std_per_group": {k: float(v.std()) for k, v in groups.items()},
        "pairwise_auroc": {},
        "per_ood_dataset": {},
        "ds_per_aug": {},
    }

    for a, b in [("ID", "OOD"), ("ID", "DS"), ("DS", "OOD"),
                 ("ID", "DS_real"), ("DS_real", "OOD"), ("DS", "DS_real")]:
        if a in groups and b in groups:
            summary["pairwise_auroc"][f"{a}_vs_{b}"] = float(auroc(groups[b], groups[a]))

    # Per-OOD-dataset breakdown
    for ds_name, ds_scores in ood_by_dataset.items():
        entry = {
            "n": int(len(ds_scores)),
            "mean": float(ds_scores.mean()),
            "std": float(ds_scores.std()),
        }
        if "ID" in groups:
            entry["auroc_vs_ID"] = float(auroc(ds_scores, groups["ID"]))
        if "DS" in groups:
            entry["auroc_vs_DS"] = float(auroc(ds_scores, groups["DS"]))
        if "DS_real" in groups:
            entry["auroc_vs_DS_real"] = float(auroc(ds_scores, groups["DS_real"]))
        summary["per_ood_dataset"][ds_name] = entry

    # DS per-aug
    for aug_name in augs.keys():
        ds_aug = np.array([r["score"] for r in records if r["group"]=="DS" and r["subgroup"]==aug_name])
        if len(ds_aug) > 0 and "ID" in groups:
            summary["ds_per_aug"][aug_name] = {
                "n": int(len(ds_aug)),
                "mean": float(ds_aug.mean()),
                "std": float(ds_aug.std()),
                "auroc_vs_ID": float(auroc(ds_aug, groups["ID"])),
            }

    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for g, v in groups.items():
        print(f"  {g:<10} n={len(v):4d}  mean={v.mean():7.3f}  std={v.std():6.3f}")
    print()
    for k, v in summary["pairwise_auroc"].items():
        print(f"  AUROC {k:<24} = {v:.3f}")
    print(f"\n  Per-OOD-dataset:")
    for ds_name, e in summary["per_ood_dataset"].items():
        line = f"    {ds_name:<14} n={e['n']:3d}  mean={e['mean']:7.3f}"
        if "auroc_vs_ID" in e: line += f"  AUROC vs ID={e['auroc_vs_ID']:.3f}"
        if "auroc_vs_DS_real" in e: line += f"  vs DS_real={e['auroc_vs_DS_real']:.3f}"
        print(line)
    print(f"\n  DS per-augmentation:")
    if "ID" in groups:
        for k, v in summary["ds_per_aug"].items():
            delta = v["mean"] - groups["ID"].mean()
            print(f"    {k:<12} mean={v['mean']:7.3f}  Δ={delta:+.3f}  AUROC vs ID = {v['auroc_vs_ID']:.3f}")

    # =======================================================================
    # Figure 1: Density-normalized histogram (all groups comparable)
    # =======================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = {"ID": "#4C72B0", "DS": "#DDA43F", "DS_real": "#9B6B9E", "OOD": "#DD4444"}
    all_scores = np.concatenate([v for v in groups.values()])
    bins = np.linspace(all_scores.min()*0.95, all_scores.max()*1.05, 35)
    for g, v in groups.items():
        ax.hist(v, bins=bins, alpha=0.55, density=True,
                label=f"{g} (n={len(v)}, μ={v.mean():.2f})",
                color=colors.get(g, "gray"), edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Median GSVD ratio", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Score distributions (density-normalized, comparable across groups)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "three_way_histogram.pdf")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved histogram → {p}")

    # =======================================================================
    # Figure 2: Pairwise ROC (now includes DS_real)
    # =======================================================================
    fig, ax = plt.subplots(figsize=(6, 5.5))
    roc_pairs = [
        ("ID",      "OOD",     "#DD4444"),
        ("ID",      "DS",      "#DDA43F"),
        ("ID",      "DS_real", "#9B6B9E"),
        ("DS",      "OOD",     "#4C72B0"),
        ("DS_real", "OOD",     "#6B8E23"),
    ]
    for neg, pos, col in roc_pairs:
        if neg in groups and pos in groups:
            y = np.concatenate([np.zeros(len(groups[neg])), np.ones(len(groups[pos]))])
            s = np.concatenate([groups[neg], groups[pos]])
            fpr, tpr, _ = roc_curve(y, s)
            a = roc_auc_score(y, s)
            ax.plot(fpr, tpr, color=col, linewidth=2, label=f"{neg} vs {pos} (AUROC={a:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Pairwise ROC: what does the score separate?", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "roc_pairwise.pdf")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC → {p}")

    # =======================================================================
    # Figure 3: Per-augmentation bars + DS_real reference
    # =======================================================================
    fig, ax = plt.subplots(figsize=(8, 4.5))
    aug_names = list(augs.keys())
    id_mean = groups["ID"].mean() if "ID" in groups else 0.0
    aug_means = [np.mean([r["score"] for r in records
                          if r["group"]=="DS" and r["subgroup"]==a]) for a in aug_names]
    aug_stds = [np.std([r["score"] for r in records
                        if r["group"]=="DS" and r["subgroup"]==a]) for a in aug_names]
    x = np.arange(len(aug_names))
    ax.bar(x, aug_means, yerr=aug_stds, color="#DDA43F", alpha=0.8,
           edgecolor="black", linewidth=0.5, capsize=4, label="DS (synthetic aug)")
    ax.axhline(id_mean, color="#4C72B0", linestyle="--", linewidth=2,
               label=f"ID mean ({id_mean:.2f})")
    if "DS_real" in groups:
        ax.axhline(groups["DS_real"].mean(), color="#9B6B9E", linestyle="--", linewidth=2,
                   label=f"DS_real mean ({groups['DS_real'].mean():.2f})")
    if "OOD" in groups:
        ax.axhline(groups["OOD"].mean(), color="#DD4444", linestyle="--", linewidth=2,
                   label=f"OOD mean ({groups['OOD'].mean():.2f})")
    ax.set_xticks(x); ax.set_xticklabels(aug_names)
    ax.set_ylabel("Median GSVD ratio", fontsize=11)
    ax.set_title("Score shift per augmentation (error bars = std)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.2, axis="y")
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "score_vs_augmentation.pdf")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-aug plot → {p}")

    # =======================================================================
    # Figure 4 (NEW): GSVD score vs mean image luminance — diagnostic scatter
    # =======================================================================
    fig, ax = plt.subplots(figsize=(8, 5.5))

    group_order = ["ID", "DS", "DS_real", "OOD"]
    pooled_lum, pooled_score = [], []

    for g in group_order:
        g_records = [r for r in records if r["group"] == g]
        if not g_records:
            continue
        lums = np.array([r["luminance"] for r in g_records])
        scores = np.array([r["score"] for r in g_records])

        # subsample DS for visual clarity (500 points is enough to see pattern)
        if g == "DS" and len(lums) > 500:
            idx = np.random.RandomState(0).choice(len(lums), 500, replace=False)
            lums_plot, scores_plot = lums[idx], scores[idx]
        else:
            lums_plot, scores_plot = lums, scores

        # per-group Pearson correlation computed on FULL data
        r_pearson = np.corrcoef(lums, scores)[0, 1] if len(lums) > 1 else float("nan")

        ax.scatter(lums_plot, scores_plot, alpha=0.35, s=14,
                   color=colors.get(g, "gray"),
                   label=f"{g} (r={r_pearson:+.2f})")

        pooled_lum.extend(lums.tolist())
        pooled_score.extend(scores.tolist())

    pooled_lum = np.array(pooled_lum)
    pooled_score = np.array(pooled_score)
    r_total = np.corrcoef(pooled_lum, pooled_score)[0, 1] if len(pooled_lum) > 1 else float("nan")

    ax.set_xlabel("Mean image luminance (0–255)", fontsize=11)
    ax.set_ylabel("Median GSVD ratio", fontsize=11)
    ax.set_title(f"Is the score just a brightness detector? Pooled Pearson r = {r_total:+.3f}",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, "score_vs_luminance.pdf")
    fig.savefig(p, dpi=300, bbox_inches="tight")
    fig.savefig(p.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved luminance scatter → {p}")

    print(f"\n{'='*70}")
    print(f"  INTERPRETATION GUIDE (updated)")
    print(f"{'='*70}")
    print(f"""
  AUROC(ID vs OOD)        — headline anomaly detection (higher = better)
  AUROC(ID vs DS)         — synthetic covariate shift robustness
                            (~0.5 = robust; <0.5 = DS scores LOWER than ID;
                             >0.7 = method conflates domain with anomaly)
  AUROC(ID vs DS_real)    — real covariate shift robustness (BDD100K)
                            this is the most important number for your
                            server-gap diagnosis
  AUROC(DS_real vs OOD)   — can the method tell real DS from anomalies?
                            if this is high, method is doing real work

  Per-OOD-dataset AUROCs  — if Fishyscapes >> AnomalyTrack, method may
                            be responding to paste artifacts rather than
                            to novel semantic content

  Score vs luminance r    — magnitude matters. |r| < 0.3 means score is
                            not reducible to brightness. |r| > 0.6 means
                            it largely IS a brightness detector.
    """)
    print(f"  All outputs saved to: {OUT_DIR}/")


if __name__ == "__main__":
    main()