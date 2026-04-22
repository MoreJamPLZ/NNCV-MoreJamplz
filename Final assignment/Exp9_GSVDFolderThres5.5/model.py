import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import SegformerForSemanticSegmentation, SegformerConfig

# Segformer model 
def get_segformer_config(n_classes=19):
    # Same config as experiment
    return SegformerConfig(
        num_channels=3,
        num_labels=n_classes,
        num_encoder_blocks=4,
        depths=[3, 6, 40, 3],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[64, 128, 320, 512],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        decoder_hidden_size=768,
        semantic_loss_ignore_index=255,
    )

# GSVD Algorithm 
def gsvd0(A, B):
    # Got this from Michiel (TU/e). It compares matrix A and matrix B.
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
    # c and s are the singular values we actually care about later
    return U, V, X, c, s

# Novice, non linear CNN
class NonlinearNovice(nn.Module):
    # This is just a basic CNN. We don't train it. 
    # It just looks at the image and spits out random, untrained features.
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


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, gsvd_threshold=6.50, gsvd_start_idx=256, gsvd_n_ratios=10):
        super().__init__()
        self.in_channels = in_channels
        self.gsvd_start_idx = int(gsvd_start_idx)
        self.gsvd_n_ratios = int(gsvd_n_ratios)

        # Saving the threshold as a buffer so it gets saved inside our .pt file
        self.register_buffer("gsvd_threshold", torch.tensor(float(gsvd_threshold)))

        # Segmentic task by segformer model 
        self.segformer = SegformerForSemanticSegmentation(get_segformer_config(n_classes))

        # Expert, Dino for OOD detection
        self.dino = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=False, num_classes=0)
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False) # Freeze it!

        # Novice, non linear CNN for OOD detection
        self.novice = NonlinearNovice(in_channels=in_channels, feat_dim=320)
        self.novice.eval()
        for p in self.novice.parameters():
            p.requires_grad_(False) 

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            print(f"Wrong number of channels! Expected {self.in_channels} but got {x.shape[1]}.")
            return None

        # Segmentation map 
        logits = self.segformer(pixel_values=x).logits 
        
        # Scale it back up to the original image size (512x512)
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)

        # OOD Detection with GSVD
        include_decisions = []
        
        for i in range(x.shape[0]):
            single_img = x[i:i+1] # Grab one image but keep it 4D: (1, C, H, W)
            
            # Get features from both models
            dumb_feats = self.novice(single_img)
            smart_feats_raw = self.dino.forward_features(single_img)
            
            # DINO adds 5 extra tokens at the start (1 CLS + 4 Registers). 
            tokens = smart_feats_raw[:, 5:, :]
            
            # Turn the flat list of tokens back into a 2D grid
            n_tok = tokens.shape[1]
            grid_size = int(round(n_tok ** 0.5))
            smart_grid = tokens.permute(0, 2, 1).reshape(1, 768, grid_size, grid_size)

            # Flatten them out for the GSVD math and normalize them
            # NumPy needs them as (Channels, Pixels)
            A = F.normalize(dumb_feats.squeeze(0).flatten(1).T, dim=1).T.cpu().numpy()
            B = F.normalize(smart_grid.squeeze(0).flatten(1).T, dim=1).T.cpu().numpy()

            # Run the math to compare them
            _, _, _, C, S = gsvd0(A, B)
            gen_sv = C / S
            
            # Look at a specific slice of the results based on our calibration
            ratios = gen_sv[self.gsvd_start_idx : self.gsvd_start_idx + self.gsvd_n_ratios]
            finite = ratios[np.isfinite(ratios)]

            # Decide if it's safe to keep (score must be lower than threshold)
            if len(finite) == 0:
                is_safe = False # Something broke in math, play it safe and reject
            else:
                score = float(np.median(finite))
                is_safe = (score <= float(self.gsvd_threshold.item()))
                
            include_decisions.append(is_safe)

        # Convert our list of True/False decisions into a PyTorch tensor
        decisions_tensor = torch.tensor(include_decisions, device=x.device)

        return logits, decisions_tensor