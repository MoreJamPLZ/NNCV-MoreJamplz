"""
OOD-aware segmentation model. 

Here, 
  1. SegFormer-B5 (pretrained) sementic segmentation head
  2. DINOv3 ViT-B/16 (pretrained, frozen) expert
  3. Nonlinear CNN (loaded from saved weights, frozen) novice

Forward returns:
  - logits: (B, 19, H/4, W/4) segmentation predictions
  - include: bool. True if image is in-distribution, False if OOD
"""

import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerForSemanticSegmentation, SegformerConfig


# Path to the frozen novice weights saved during threshold calibration.
NOVICE_WEIGHTS_PATH = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/weights/novice_nonlinear_seed1.pt"

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
# Novice: Nonlinear CNN (weights loaded from disk, frozen)
# ============================================================================
class NonlinearNovice(nn.Module):
    """4 stride-2 conv+BN+ReLU blocks (3→64→128→256→320). ~1.2M params."""
    def __init__(self, in_channels=3, feat_dim=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================================
# Main Model
# ============================================================================
class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels

        # Runtime OOD mode can be overridden for fast benchmarking:
        #   OOD_METHOD in {gsvd, energy, maxlogit, entropy}
        #   OOD_THRESHOLD as float (optional)
        self.ood_method = os.getenv("OOD_METHOD", "gsvd").strip().lower()
        self.ood_threshold_override = os.getenv("OOD_THRESHOLD")
        valid_methods = {"gsvd", "energy", "maxlogit", "entropy"}
        if self.ood_method not in valid_methods:
            self.ood_method = "gsvd"

        # --- Segmentation head: SegFormer-B5 ---
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

        # --- OOD expert/novice (only needed for GSVD mode) ---
        self.dino = None
        self.novice = None
        if self.ood_method == "gsvd":
            self.dino = timm.create_model(
                "vit_base_patch16_dinov3.lvd1689m",
                pretrained=False,
                num_classes=0,
            )

            # Novice: load the exact weights used during threshold calibration.
            # Seeding would also work in principle, but saved weights are
            # immune to PyTorch/CUDA version drift and construction-order
            # effects on the RNG. The threshold below was calibrated against
            # *these specific* weights.
            self.novice = NonlinearNovice(in_channels=in_channels, feat_dim=320)
            if not os.path.exists(NOVICE_WEIGHTS_PATH):
                raise FileNotFoundError(
                    f"Novice weights not found at {NOVICE_WEIGHTS_PATH}. "
                    "This file must be shipped alongside model.py, since the "
                    "OOD threshold was calibrated against these specific "
                    "random-init weights."
                )
            state = torch.load(NOVICE_WEIGHTS_PATH, map_location="cpu",
                               weights_only=True)
            missing, unexpected = self.novice.load_state_dict(state, strict=True)
            # strict=True raises on mismatch; these lists will be empty.
            self.novice.eval()
            for p in self.novice.parameters():
                p.requires_grad_(False)

        # --- GSVD hyperparameters (from ablation studies) ---
        self.register_buffer("ood_threshold", torch.tensor(5.4385))  # n=20, median
        self.gsvd_start_idx = 256
        self.gsvd_n_ratios = 10

        # --- Logit-based OOD defaults (tune with tune_threshold.py) ---
        self.register_buffer("energy_threshold", torch.tensor(-2.5))
        self.register_buffer("maxlogit_threshold", torch.tensor(-6.0))
        self.register_buffer("entropy_threshold", torch.tensor(2.0))

    def forward(self, x):
        # 1. Segmentation
        seg_out = self.segformer(pixel_values=x)
        logits = seg_out.logits

        # 2. OOD detection
        include = self._detect_ood(x, logits)

        return logits, include

    def _active_threshold(self):
        if self.ood_threshold_override is not None:
            try:
                return float(self.ood_threshold_override)
            except ValueError:
                pass

        if self.ood_method == "entropy":
            return float(self.entropy_threshold.item())
        return float(self.ood_threshold.item())

    @torch.no_grad()
    def _logit_ood_score(self, logits):
        probs = torch.softmax(logits.float(), dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
        return float(entropy.mean().item())

    @torch.no_grad()
    def _detect_ood(self, x, logits):
        """Returns True if in-distribution, False if OOD."""
        if self.ood_method in {"energy", "maxlogit", "entropy"}:
            score = self._logit_ood_score(logits)
            return score <= self._active_threshold()

        if self.dino is None or self.novice is None:
            return False

        # Novice features
        A_feat = self.novice(x)  # (1, 320, 32, 32)

        # Expert features
        dino_out = self.dino.forward_features(x)
        B_tokens = dino_out[:, 5:, :]  # skip register/CLS tokens
        N = B_tokens.shape[1]
        h = w = int(N ** 0.5)
        B_grid = B_tokens.permute(0, 2, 1).reshape(1, 768, h, w)
        B_feat = F.interpolate(
            B_grid, size=(A_feat.shape[2], A_feat.shape[3]),
            mode="bilinear", align_corners=False,
        )

        # Flatten and normalize
        A = A_feat.squeeze(0).flatten(1).T  # (HW, 320)
        B = B_feat.squeeze(0).flatten(1).T  # (HW, 768)
        A_norm = F.normalize(A, dim=1).T.cpu().numpy()  # (320, HW)
        B_norm = F.normalize(B, dim=1).T.cpu().numpy()  # (768, HW)

        # GSVD
        _, _, _, C, S = gsvd0(A_norm, B_norm)
        gen_sv = C / S
        ratios = gen_sv[self.gsvd_start_idx : self.gsvd_start_idx + self.gsvd_n_ratios]
        finite = ratios[np.isfinite(ratios)]

        if len(finite) == 0:
            return False

        score = float(np.median(finite))
        # Higher score → more OOD-like. include = (score is low enough).
        return score <= self._active_threshold()