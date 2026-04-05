from cv2 import transform
from matplotlib.colors import Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Dinov2Model, SegformerForSemanticSegmentation, SegformerConfig
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode
import os
import glob
import csv
from datetime import datetime

def gsvd0(A, B):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    M = np.vstack([A, B])
    Q, R = np.linalg.qr(M, mode='reduced')
    m1 = A.shape[0]
    Q1, Q2 = Q[:m1, :], Q[m1:, :]
    U, c, Wt = np.linalg.svd(Q1, full_matrices=False)
    W = Wt.T
    s = np.sqrt(np.maximum(1.0 - c**2, 0.0))
    Q2W = Q2 @ W
    V = np.zeros_like(Q2W)
    for i in range(Q2W.shape[1]):
        nrm = np.linalg.norm(Q2W[:, i])
        if nrm > 1e-14:
            V[:, i] = Q2W[:, i] / nrm
    X = R.T @ W
    return U, V, X, c, s


def deim(U, k):
    indices = np.zeros(k, dtype=int)
    indices[0] = np.argmax(np.abs(U[:, 0]))
    for j in range(1, k):
        P = indices[:j]
        coeff = np.linalg.solve(U[np.ix_(P, np.arange(j))], U[P, j])
        r = U[:, j] - U[:, :j] @ coeff
        indices[j] = np.argmax(np.abs(r))
    return indices

class DeepRandomProjection(nn.Module):
    def __init__(self, in_channels=3, out_channels=768, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        
        self.layers = nn.Sequential(
            # Stage 1: 512 -> 128
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3),
            nn.GroupNorm(1, 64),
            nn.GELU(),
            
            # Stage 2: 128 -> 64
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 128),
            nn.GELU(),
            
            # Stage 3: 64 -> 32
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(1, 320),
            nn.GELU(),
            
            # Stage 4: 32 -> 32 (stride=1, no spatial reduction)
            nn.Conv2d(320, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.layers(x)
    
# class FixedRandomProjection(nn.Module):
#     def __init__(self, in_channels=3, out_channels=320, patch_size=16, seed=42):
#         super().__init__()
#         torch.manual_seed(seed)
#         # Single conv layer = random patch-wise projection
#         self.proj = nn.Conv2d(in_channels, out_channels, 
#                               kernel_size=patch_size, stride=patch_size)
#         for p in self.parameters():
#             p.requires_grad_(False)
            
#     def forward(self, x):
#         return self.proj(x)  # (1, 320, 32, 32) for 512×512 input

# class DeepRandomProjection(nn.Module):
#     def __init__(self, in_channels=3, out_channels=320, seed=42):
#         super().__init__()
#         torch.manual_seed(seed)
#         self.layers = nn.Sequential(
#             nn.Conv2d(in_channels, 64, kernel_size=4, stride=4),
#             nn.GELU(),
#             nn.Conv2d(64, 128, kernel_size=2, stride=2),
#             nn.GELU(),
#             nn.Conv2d(128, out_channels, kernel_size=2, stride=2),
#             nn.GELU(),
#         )  # 512 → 128 → 64 → 32, so output is (1, 320, 32, 32)
#         for p in self.parameters():
#             p.requires_grad_(False)

#     def forward(self, x):
#         return self.layers(x)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. Replace Segformer with the Random Projection module
        # self.random_proj = FixedRandomProjection(in_channels=3, out_channels=320, patch_size=16)
        # self.random_proj = DeepRandomProjection(in_channels=3, out_channels=320, seed=42)  # Alternative: deeper random projection
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        self.random_DINO_projection = DeepRandomProjection(in_channels=3, out_channels=768, seed=42)  # Match DINO's 768-dim features
  

    def forward(self, x):
        seg_outputs = self.segformer(pixel_values=x, output_hidden_states=True)
        logits = seg_outputs.logits
        A_spatial = seg_outputs.hidden_states[2]
        # 2. Get spatial features from the random projection instead of Segformer
        B_spatial = self.random_DINO_projection(x)
        A = A_spatial.squeeze(0).flatten(1).T
        B = B_spatial.squeeze(0).flatten(1).T

        A_norm = F.normalize(A, dim=1).T.cpu().numpy()
        B_norm = F.normalize(B, dim=1).T.cpu().numpy()

        U, V, X, C, S = gsvd0(A_norm, B_norm)
        gen_sv = C / S

        # Return ALL 64 finite ratios — we'll slice in the sweep loop
        start_idx = 256
        all_ratios = gen_sv[start_idx:start_idx + 64]

        # Return None for logits so `logits, all_ratios = model(...)` doesn't break
        return None, all_ratios

def find_optimal_threshold(id_scores, ood_scores):
    """Sweep thresholds to maximize accuracy. Returns (threshold, accuracy, tp, fp, fn, tn)."""
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])

    sorted_scores = np.sort(np.unique(scores))
    candidates = (sorted_scores[:-1] + sorted_scores[1:]) / 2

    best_acc, best_t = 0, 0
    best_tp = best_fp = best_fn = best_tn = 0

    for t in candidates:
        preds = (scores > t).astype(int)
        acc = (preds == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_t = t
            best_tp = int(((preds == 1) & (labels == 1)).sum())
            best_fp = int(((preds == 1) & (labels == 0)).sum())
            best_fn = int(((preds == 0) & (labels == 1)).sum())
            best_tn = int(((preds == 0) & (labels == 0)).sum())

    return best_t, best_acc, best_tp, best_fp, best_fn, best_tn


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"gsvd_sweep_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory: {out_dir}")

    model = Model()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    preprocess = Compose([
        ToImage(),
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    base = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz"

    datasets = {
        "ID": [
            f"{base}/data/cityscapes/leftImg8bit/val/tubingen",
            # f"{base}/data/cityscapes/leftImg8bit/val/ulm",
            # f"{base}/data/cityscapes/leftImg8bit/val/weimar",
            # f"{base}/data/cityscapes/leftImg8bit/val/zurich",
        ],
        "OOD": [
            f"{base}/fishyscapes_rgb_100",
            # f"{base}/dataset_AnomalyTrack/images",
        ],
    }

    # --- Step 1: Run inference once, store all 64 ratios per image ---
    raw_results = {"ID": [], "OOD": []}

    for label, folders in datasets.items():
        paths = []
        for folder in folders:
            paths += sorted(glob.glob(os.path.join(folder, "*.png")))
            paths += sorted(glob.glob(os.path.join(folder, "*.jpg")))

        print(f"\n{'='*60}")
        print(f"  {label}: {len(paths)} images")
        print(f"{'='*60}")

        for i, path in enumerate(paths):
            try:
                raw_img = Image.open(path).convert("RGB")
                img_tensor = preprocess(raw_img).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits, all_ratios = model(img_tensor)

                raw_results[label].append({
                    "path": path,
                    "filename": os.path.basename(path),
                    "all_ratios": all_ratios.copy(),
                })

                finite = all_ratios[np.isfinite(all_ratios)]
                print(f"  [{i+1:3d}/{len(paths)}] mean={finite.mean():5.2f}  max={finite.max():7.2f}  {os.path.basename(path)}")

            except Exception as e:
                print(f"  [{i+1:3d}/{len(paths)}] FAILED: {os.path.basename(path)} — {e}")

    # --- Step 2: Save per-image results (all 64 ratios) ---
    per_image_csv = os.path.join(out_dir, "per_image_ratios.csv")
    with open(per_image_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label", "filename"] + [f"ratio_{j+1}" for j in range(64)]
        writer.writerow(header)
        for label in ["ID", "OOD"]:
            for entry in raw_results[label]:
                row = [label, entry["filename"]] + [f"{r:.6f}" for r in entry["all_ratios"]]
                writer.writerow(row)
    print(f"\nSaved per-image ratios → {per_image_csv}")

    # --- Step 3: Sweep over number of ratios used ---
    n_ratios_list = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 64]
    sweep_rows = []

    print(f"\n{'='*60}")
    print("  SWEEP: varying number of ratios")
    print(f"{'='*60}")
    print(f"  {'n_ratios':>8}  {'metric':>6}  {'threshold':>10}  {'accuracy':>8}  {'TP':>4}  {'FP':>4}  {'FN':>4}  {'TN':>4}  {'precision':>9}  {'recall':>6}")
    print(f"  {'-'*80}")

    for n_ratios in n_ratios_list:
        # Compute mean and max over the first n_ratios for each image
        id_means, id_maxs = [], []
        ood_means, ood_maxs = [], []

        for entry in raw_results["ID"]:
            subset = entry["all_ratios"][:n_ratios]
            finite = subset[np.isfinite(subset)]
            if len(finite) > 0:
                id_means.append(finite.mean())
                id_maxs.append(finite.max())

        for entry in raw_results["OOD"]:
            subset = entry["all_ratios"][:n_ratios]
            finite = subset[np.isfinite(subset)]
            if len(finite) > 0:
                ood_means.append(finite.mean())
                ood_maxs.append(finite.max())

        id_means = np.array(id_means)
        id_maxs = np.array(id_maxs)
        ood_means = np.array(ood_means)
        ood_maxs = np.array(ood_maxs)

        # Find best threshold for both metrics
        for metric_name, id_vals, ood_vals in [("mean", id_means, ood_means), ("max", id_maxs, ood_maxs)]:
            thresh, acc, tp, fp, fn, tn = find_optimal_threshold(id_vals, ood_vals)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            sweep_rows.append({
                "n_ratios": n_ratios,
                "metric": metric_name,
                "threshold": thresh,
                "accuracy": acc,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision": prec,
                "recall": rec,
                "id_mean": id_vals.mean(),
                "id_std": id_vals.std(),
                "ood_mean": ood_vals.mean(),
                "ood_std": ood_vals.std(),
            })

            print(f"  {n_ratios:>8}  {metric_name:>6}  {thresh:>10.4f}  {acc:>7.1%}  {tp:>4}  {fp:>4}  {fn:>4}  {tn:>4}  {prec:>8.1%}  {rec:>5.1%}")

    # --- Step 4: Save sweep results CSV ---
    sweep_csv = os.path.join(out_dir, "sweep_results.csv")
    with open(sweep_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sweep_rows[0].keys())
        writer.writeheader()
        writer.writerows(sweep_rows)
    print(f"\nSaved sweep results → {sweep_csv}")

    # --- Step 5: Plot accuracy vs n_ratios ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric_name, color, marker in [("mean", "tab:blue", "o"), ("max", "tab:red", "s")]:
        rows = [r for r in sweep_rows if r["metric"] == metric_name]
        ns = [r["n_ratios"] for r in rows]
        accs = [r["accuracy"] for r in rows]
        ax.plot(ns, accs, color=color, marker=marker, label=f"{metric_name} ratio", linewidth=2)
    ax.set_xlabel("Number of ratios used")
    ax.set_ylabel("Accuracy")
    ax.set_title("OOD Detection Accuracy vs. Number of GSVD Ratios")
    ax.set_xticks(n_ratios_list)
    ax.set_ylim(0.5, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "accuracy_vs_nratios.png"), dpi=150)
    plt.show()
    print(f"Saved → accuracy_vs_nratios.png")

    # --- Step 6: Plot histograms for each n_ratios ---
    hist_dir = os.path.join(out_dir, "histograms")
    os.makedirs(hist_dir, exist_ok=True)

    for n_ratios in n_ratios_list:
        id_means = np.array([
            entry["all_ratios"][:n_ratios][np.isfinite(entry["all_ratios"][:n_ratios])].mean()
            for entry in raw_results["ID"]
        ])
        ood_means = np.array([
            entry["all_ratios"][:n_ratios][np.isfinite(entry["all_ratios"][:n_ratios])].mean()
            for entry in raw_results["OOD"]
        ])
        id_maxs = np.array([
            entry["all_ratios"][:n_ratios][np.isfinite(entry["all_ratios"][:n_ratios])].max()
            for entry in raw_results["ID"]
        ])
        ood_maxs = np.array([
            entry["all_ratios"][:n_ratios][np.isfinite(entry["all_ratios"][:n_ratios])].max()
            for entry in raw_results["OOD"]
        ])

        # Get thresholds for this n_ratios
        mean_row = next(r for r in sweep_rows if r["n_ratios"] == n_ratios and r["metric"] == "mean")
        max_row = next(r for r in sweep_rows if r["n_ratios"] == n_ratios and r["metric"] == "max")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"GSVD Ratio Distributions — using top {n_ratios} ratios", fontsize=14)

        axes[0].hist(id_means, bins=20, alpha=0.6, label="ID", color="tab:blue")
        axes[0].hist(ood_means, bins=20, alpha=0.6, label="OOD", color="tab:red")
        axes[0].axvline(mean_row["threshold"], color="k", linestyle="--",
                        label=f"threshold={mean_row['threshold']:.2f} (acc={mean_row['accuracy']:.1%})")
        axes[0].set_xlabel("Mean ratio")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Mean ratio")
        axes[0].legend()

        axes[1].hist(id_maxs, bins=20, alpha=0.6, label="ID", color="tab:blue")
        axes[1].hist(ood_maxs, bins=20, alpha=0.6, label="OOD", color="tab:red")
        axes[1].axvline(max_row["threshold"], color="k", linestyle="--",
                        label=f"threshold={max_row['threshold']:.2f} (acc={max_row['accuracy']:.1%})")
        axes[1].set_xlabel("Max ratio")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Max ratio")
        axes[1].legend()

        plt.tight_layout()
        fname = f"hist_n{n_ratios:02d}.png"
        fig.savefig(os.path.join(hist_dir, fname), dpi=150)
        plt.close(fig)

    print(f"Saved {len(n_ratios_list)} histograms → {hist_dir}/")
    print(f"\nAll outputs saved in: {out_dir}/")