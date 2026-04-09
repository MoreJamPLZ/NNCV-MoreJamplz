"""
OOD-aware segmentation model for challenge submission.

Components:
  1. SegFormer-B5 (pretrained) — semantic segmentation logits
  2. DINOv3 ViT-B/16 (pretrained, frozen) — expert features for GSVD
  3. Nonlinear CNN (randomly initialized, frozen) — novice features for GSVD

Forward returns:
  - logits: (B, 19, H/4, W/4) segmentation predictions
  - include: bool — True if image is in-distribution, False if OOD
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerForSemanticSegmentation, SegformerConfig


# ============================================================================
# GSVD (runs on CPU/numpy)
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
# Novice: Nonlinear CNN (randomly initialized)
# ============================================================================
class NonlinearNovice(nn.Module):
    def __init__(self, in_channels=3, feat_dim=320):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
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

        # --- OOD expert: DINOv3 ViT-B/16 (frozen) ---
        self.dino = timm.create_model(
            "vit_base_patch16_dinov3.lvd1689m",
            pretrained=False,
            num_classes=0,
        )

        # --- OOD novice: Nonlinear CNN (frozen) ---
        self.novice = NonlinearNovice(in_channels=in_channels, feat_dim=320)

        # --- GSVD hyperparameters (from ablation studies) ---
        # UPDATE THESE after your final validation experiments
        self.register_buffer("ood_threshold", torch.tensor(5.4385))  # from ablation 3, n=20
        self.gsvd_start_idx = 256
        self.gsvd_n_ratios = 20

    def forward(self, x):
        # 1. Segmentation
        seg_out = self.segformer(pixel_values=x)
        logits = seg_out.logits

        # 2. OOD detection via GSVD
        include = self._detect_ood(x)

        return logits, include

    @torch.no_grad()
    def _detect_ood(self, x):
        """Returns True if in-distribution, False if OOD."""
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
            return False  # can't determine, flag as OOD

        score = float(np.median(finite))
        # OOD images have HIGHER scores → if score > threshold, it's OOD
        return score <= self.ood_threshold.item()
