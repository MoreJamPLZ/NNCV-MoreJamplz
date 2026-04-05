import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

# Transformers & Vision Models
from transformers import Dinov2Model, SegformerForSemanticSegmentation, SegformerConfig
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode

# ============================================================================
# Generalized Singular Value Decomposition (GSVD)
# ============================================================================
#
# Given two matrices A (m1 x n) and B (m2 x n) sharing the same column space 
# (e.g. same image, different feature extractors),the GSVD finds decompositions 
# such that:
#
#     A = U · diag(C) · X^T
#     B = V · diag(S) · X^T
#
# where C and S satisfy C^2 + S^2 = I (analogous to cos/sin).
#
# The generalized singular values are the ratios  c_i / s_i.
# These ratios indicate how much each direction is "explained" by A vs B:
#   - Large  c/s  → direction is dominant in A (SegFormer)
#   - Small  c/s  → direction is dominant in B (DINOv2)
#   - c/s ≈ 1     → direction is equally shared
#
# This is used here to compare the feature subspaces of SegFormer and DINOv2:
# a shift in the generalized singular value spectrum between in-distribution
# and out-of-distribution images can serve as an OOD detection signal.
# ============================================================================
 

def gsvd0(A: np.ndarray, B: np.ndarray):
    """
    Computes a reduced Generalized Singular Value Decomposition (GSVD) for matrices A and B (Paige & Saunders, 1981).
    This helps find the generalized principal angles between two feature spaces.  

    Parameters
    ----------
    A : array, shape (m1, n) — e.g. (320, 1024) from SegFormer features. 1024 = 32x32 spatial tokens, 320 = feature dim
    B : array, shape (m2, n) — e.g. (768, 1024) from DINOv2 features. 
 
    Returns
    -------
    U : (m1, k) — left singular vectors for A    (SegFormer subspace basis)
    V : (m2, k) — left singular vectors for B    (DINOv2 subspace basis)
    X : (n, k)  — shared right singular vectors  (spatial directions)
    C : (k,)    — cosine-like singular values for A  (0 ≤ c_i ≤ 1)
    S : (k,)    — sine-like singular values for B    (0 ≤ s_i ≤ 1, c² + s² = 1)
    """

    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    # Compute joint QR factorisation of M = [A; B]
    M = np.vstack([A, B])  # shape (m1+m2, n). m1 = 320 (SegFormer), m2 = 768 (DINOv2)
    Q, R = np.linalg.qr(M, mode='reduced') # shape Q=(m1+m2, k), R=(k, n) where k = min(m1+m2, n)

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


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()

        # self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b4-finetuned-cityscapes-1024-1024")
        self.segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
        # config = SegformerConfig(
        #     num_channels=in_channels, num_labels=n_classes,
        #     num_encoder_blocks=4, depths=[3, 6, 40, 3],
        #     sr_ratios=[8, 4, 2, 1], hidden_sizes=[64, 128, 320, 512],
        #     num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
        #     hidden_act="gelu", hidden_dropout_prob=0.0,
        #     attention_probs_dropout_prob=0.0, classifier_dropout_prob=0.1,
        #     decoder_hidden_size=768, semantic_loss_ignore_index=255,
        # )
        # self.segformer = SegformerForSemanticSegmentation(config)
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # -- SegFormer --
        seg_outputs = self.segformer(pixel_values=x, output_hidden_states=True)
        logits = seg_outputs.logits
        A_spatial = seg_outputs.hidden_states[2]

        # -- DINOv2 --
        dino_outputs = self.dino(pixel_values=x)
        B_tokens = dino_outputs.last_hidden_state[:, 1:, :]
        N = B_tokens.shape[1]
        h = w = int(N ** 0.5)
        B_grid = B_tokens.permute(0, 2, 1).reshape(1, 768, h, w)
        B_aligned = F.interpolate(
            B_grid, size=(A_spatial.shape[2], A_spatial.shape[3]),
            mode="bilinear", align_corners=False,
        )

        A = A_spatial.squeeze(0).flatten(1).T
        B = B_aligned.squeeze(0).flatten(1).T

        A_norm = F.normalize(A, dim=1).T.cpu().numpy()   # (320, 1024)
        B_norm = F.normalize(B, dim=1).T.cpu().numpy()   # (768, 1024)

        U, V, X, C, S = gsvd0(A_norm, B_norm)

        gen_sv = C / S
        start_idx = 256
        k = 2
        ratios = gen_sv[start_idx:start_idx + 10]

        # U_sel = U[:, start_idx:start_idx + k]
        # V_sel = V[:, start_idx:start_idx + k]

        # indU = deim(U_sel, k)
        # indV = deim(V_sel, k)

        # Ap = A_norm[indU, :]
        # Bp = B_norm[indV, :]
        # Aq = U_sel.T @ A_norm
        # Bq = V_sel.T @ B_norm

        # # Visualize
        # ima = Ap[0, :]
        # if np.sum(ima) < 0:
        #     ima = -ima
        # ima = ima.reshape(32, 32)

        # imb = Bp[0, :]
        # if np.sum(imb) < 0:
        #     imb = -imb
        # imb = imb.reshape(32, 32)

        return logits, ratios


# ==========================================
# TEST SCRIPT — iterate over all images, compute boundary
# ==========================================
if __name__ == "__main__":
    print("Initializing model...")
    model = Model(in_channels=3, n_classes=19)
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
            f"{base}/data/cityscapes/leftImg8bit/val/ulm",
            f"{base}/data/cityscapes/leftImg8bit/val/weimar",
            f"{base}/data/cityscapes/leftImg8bit/val/zurich",
        ],
        "OOD": [
            f"{base}/fishyscapes_rgb_100",
            f"{base}/dataset_AnomalyTrack/images",
        ],
    }

    # Collect results
    results = {"ID": [], "OOD": []}

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
                    logits, ratios = model(img_tensor)

                finite_ratios = ratios[np.isfinite(ratios)]
                max_r = finite_ratios.max()
                mean_r = finite_ratios.mean()
                median_r = np.median(finite_ratios)

                results[label].append({
                    "path": path,
                    "max": max_r,
                    "mean": mean_r,
                    "median": median_r,
                })

                print(f"  [{i+1:3d}/{len(paths)}] max={max_r:7.2f}  mean={mean_r:5.2f}  median={median_r:5.2f}  {os.path.basename(path)}")

            except Exception as e:
                print(f"  [{i+1:3d}/{len(paths)}] FAILED: {os.path.basename(path)} — {e}")

    # ==========================================
    # Summary statistics
    # ==========================================
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")

    id_means = np.array([r["mean"] for r in results["ID"]])
    ood_means = np.array([r["mean"] for r in results["OOD"]])
    id_maxs = np.array([r["max"] for r in results["ID"]])
    ood_maxs = np.array([r["max"] for r in results["OOD"]])

    print(f"\n  ID  images: {len(id_means)}")
    print(f"    mean ratio — min={id_means.min():.2f}  max={id_means.max():.2f}  avg={id_means.mean():.2f}  std={id_means.std():.2f}")
    print(f"    max  ratio — min={id_maxs.min():.2f}  max={id_maxs.max():.2f}  avg={id_maxs.mean():.2f}  std={id_maxs.std():.2f}")

    print(f"\n  OOD images: {len(ood_means)}")
    print(f"    mean ratio — min={ood_means.min():.2f}  max={ood_means.max():.2f}  avg={ood_means.mean():.2f}  std={ood_means.std():.2f}")
    print(f"    max  ratio — min={ood_maxs.min():.2f}  max={ood_maxs.max():.2f}  avg={ood_maxs.mean():.2f}  std={ood_maxs.std():.2f}")

    # ==========================================
    # Find optimal threshold (maximize accuracy)
    # ==========================================
    all_scores = np.concatenate([id_means, ood_means])
    all_labels = np.concatenate([np.zeros(len(id_means)), np.ones(len(ood_means))])  # 0=ID, 1=OOD

    best_acc = 0
    best_thresh = 0
    best_metric = "mean"

    for metric_name, id_vals, ood_vals in [("mean", id_means, ood_means), ("max", id_maxs, ood_maxs)]:
        scores = np.concatenate([id_vals, ood_vals])
        labels = all_labels

        # Try thresholds at midpoints between sorted scores
        sorted_scores = np.sort(np.unique(scores))
        candidates = (sorted_scores[:-1] + sorted_scores[1:]) / 2

        for t in candidates:
            preds = (scores > t).astype(int)
            acc = (preds == labels).mean()
            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            tn = ((preds == 0) & (labels == 0)).sum()

            if acc > best_acc:
                best_acc = acc
                best_thresh = t
                best_metric = metric_name
                best_tp, best_fp, best_fn, best_tn = tp, fp, fn, tn

    print(f"\n  OPTIMAL THRESHOLD")
    print(f"    Metric:    {best_metric} ratio")
    print(f"    Threshold: {best_thresh:.4f}")
    print(f"    Accuracy:  {best_acc:.1%}")
    print(f"    TP={best_tp}  FP={best_fp}  FN={best_fn}  TN={best_tn}")
    print(f"    Precision: {best_tp/(best_tp+best_fp):.1%}" if (best_tp+best_fp) > 0 else "")
    print(f"    Recall:    {best_tp/(best_tp+best_fn):.1%}" if (best_tp+best_fn) > 0 else "")

    # ==========================================
    # Plot distributions
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(id_means, bins=20, alpha=0.6, label="ID", color="tab:blue")
    axes[0].hist(ood_means, bins=20, alpha=0.6, label="OOD", color="tab:red")
    if best_metric == "mean":
        axes[0].axvline(best_thresh, color="k", linestyle="--", label=f"threshold={best_thresh:.2f}")
    axes[0].set_xlabel("Mean ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Mean GSVD ratio distribution")
    axes[0].legend()

    axes[1].hist(id_maxs, bins=20, alpha=0.6, label="ID", color="tab:blue")
    axes[1].hist(ood_maxs, bins=20, alpha=0.6, label="OOD", color="tab:red")
    if best_metric == "max":
        axes[1].axvline(best_thresh, color="k", linestyle="--", label=f"threshold={best_thresh:.2f}")
    axes[1].set_xlabel("Max ratio")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Max GSVD ratio distribution")
    axes[1].legend()

    plt.tight_layout()
    plt.show()