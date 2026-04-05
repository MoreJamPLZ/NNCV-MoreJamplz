from cv2 import transform
from matplotlib.colors import Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.linalg import svd as scipy_svd
from transformers import Dinov2Model, SegformerForSemanticSegmentation, SegformerConfig
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode


def gsvd0(A, B):
    """
    GSVD via QR + CSD, matching MATLAB's gsvd0.
    Returns U, V, X, c, s (vectors) with A = U @ diag(c) @ X.T,
    B = V @ diag(s) @ X.T, c**2 + s**2 = 1, sorted so c/s is nonincreasing.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    m1, n = A.shape
    m2 = B.shape[0]

    # QR of stacked matrix
    M = np.vstack([A, B])
    Q, R = np.linalg.qr(M, mode='reduced')  # Q: (m1+m2)xk, R: kxn
    Q1, Q2 = Q[:m1, :], Q[m1:, :]

    # CSD via SVD of Q1
    U, c, Wt = np.linalg.svd(Q1, full_matrices=False)
    W = Wt.T
    s = np.sqrt(np.maximum(1.0 - c**2, 0.0))

    # V from Q2 @ W = V @ diag(s)
    Q2W = Q2 @ W
    V = np.zeros_like(Q2W)
    for i in range(Q2W.shape[1]):
        nrm = np.linalg.norm(Q2W[:, i])
        if nrm > 1e-14:
            V[:, i] = Q2W[:, i] / nrm

    # X: A = U diag(c) W' R => X' = W' R => X = R' W
    X = R.T @ W

    return U, V, X, c, s

def deim(U, k):
    """Discrete Empirical Interpolation Method — select k pivot indices."""
    indices = np.zeros(k, dtype=int)
    indices[0] = np.argmax(np.abs(U[:, 0]))
    for j in range(1, k):
        P = indices[:j]
        coeff = np.linalg.solve(U[np.ix_(P, np.arange(j))], U[P, j])
        r = U[:, j] - U[:, :j] @ coeff
        indices[j] = np.argmax(np.abs(r))
    return indices

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19, gsvd_k=50, threshold=0.5):
        super().__init__()
        self.in_channels = in_channels
        self.gsvd_k      = gsvd_k       
        self.threshold   = threshold 

        # 1. Init SegFormer
        config = SegformerConfig(
            num_channels=in_channels,
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
        self.segformer = SegformerForSemanticSegmentation(config)

        # 2. Init DINOv2
        self.dino = Dinov2Model.from_pretrained("facebook/dinov2-base")
        self.dino.eval()
        for p in self.dino.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # -- SegFormer Pass --
        seg_outputs = self.segformer(pixel_values=x, output_hidden_states=True)
        logits = seg_outputs.logits
        A_spatial = seg_outputs.hidden_states[2]   
        
        # -- DINOv2 Pass --  
        dino_outputs = self.dino(pixel_values=x)
        B_tokens = dino_outputs.last_hidden_state[:, 1:, :]  # drop CLS → (1, N, 768)

        N = B_tokens.shape[1]          # how many patch tokens?
        h = w = int(N ** 0.5)          # compute grid size from N

        B_grid = B_tokens.permute(0, 2, 1).reshape(1, 768, h, w)

        B_aligned = F.interpolate(
            B_grid,
            size=(A_spatial.shape[2], A_spatial.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        # Flatten spatial dimensions and transpose to get (n_patches, channels)
        # A_spatial: (1, 320, H', W') → (n_patches, 320) SegFormer features are 320-dim at this layer
        # B_aligned: (1, 768, H', W') → (n_patches, 768) DinOv2 features are 768-dim
        A = A_spatial.squeeze(0).flatten(1).T
        B = B_aligned.squeeze(0).flatten(1).T

        print(f"Inside forward -> A shape: {A.shape}")
        print(f"Inside forward -> B shape: {B.shape}")  
        # Normalize features to unit length
        A_norm = F.normalize(A, dim=1).T   # (n_patches, 320)
        B_norm = F.normalize(B, dim=1).T   # (n_patches, 768)

        A_norm = F.normalize(A, dim=1).T.cpu().numpy()   # (320, 1024) NumPy
        B_norm = F.normalize(B, dim=1).T.cpu().numpy()   # (768, 1024) NumPy


        U, V, X, C, S = gsvd0(A_norm, B_norm)

        gen_sv = C / S   # Generalized singular values (some may be inf or 0)

        start_idx = 256
        end_idx = 256 + 64
        k = 2

        # print(gen_sv[start_idx:end_idx])  # print the 64 gen svs in this block

        # 1. Slice the rows of U and V to restrict the search space
        # slicing columns 256:257 (the first 2 finite gen. singular values)
        U_sel = U[:, start_idx:start_idx+k]   # (320, 2) — all features, 2 gen. sing. vectors
        V_sel = V[:, start_idx:start_idx+k]   # (768, 2)

        # 3. Shift the local indices back to the global scope (256 to 319)
        # Keep these as 1D integer arrays (shape (2,)) for Python slicing!
        indU = deim(U_sel, k)   # values in 0..319 → valid rows of A_norm (320 × 1024)
        indV = deim(V_sel, k)   # values in 0..767 → valid rows of B_norm (768 × 1024)

        # If you want to print the strict MATLAB equivalents (1x2 double):
        indU_matlab = indU.reshape(1, 2).astype(np.float64)
        indV_matlab = indV.reshape(1, 2).astype(np.float64)
        print(f"MATLAB-style indU: shape {indU_matlab.shape}, type {indU_matlab.dtype}")
        print(f"MATLAB-style indV: shape {indV_matlab.shape}, type {indV_matlab.dtype}")

        # Select rows (features) picked by DEIM
        # CRITICAL: We use the 1D integer 'indU', NOT the float 'indU_matlab'
        Ap = A_norm[indU, :]    # (2, 1024) 
        Bp = B_norm[indV, :]    # (2, 1024)

        # Project onto gen. singular vectors
        Aq = U_sel.T @ A_norm   # (2, 1024)
        Bq = V_sel.T @ B_norm   # (2, 1024)

        # --- PRINT ALL SHAPES FOR COMPARISON WITH MATLAB IMAGE ---
        print("\n=== VARIABLE SHAPE CHECK ===")
        print(f"A:   {A_norm.shape} (Expected: 320x1024)")
        print(f"B:   {B_norm.shape} (Expected: 768x1024)")
        print(f"U:   {U_sel.shape}   (Expected: 320x2)")
        print(f"V:   {V_sel.shape}   (Expected: 768x2)")
        print(f"Ap:  {Ap.shape}   (Expected: 2x1024)")
        print(f"Bp:  {Bp.shape}   (Expected: 2x1024)")
        print(f"Aq:  {Aq.shape}   (Expected: 2x1024)")
        print(f"Bq:  {Bq.shape}   (Expected: 2x1024)")

        # Visualize first selected feature from A
        # Wait: Ap is (2, 1024). We take row 0 -> (1024,). 
        # For reshape(32, 32), we must use the PyTorch tensor or NumPy array.
        # Since A_norm is a PyTorch tensor, Ap is a tensor. We use .cpu().numpy() to plot.
        ima = Ap[0, :]
        if np.sum(ima) < 0:
            ima = -ima
        ima = ima.reshape(32, 32)

        print(f"ima: {ima.shape}     (Expected: 32x32)")

        plt.figure()
        plt.imshow(ima, aspect='equal', cmap='viridis')
        plt.colorbar()
        plt.title("Ap feature 1")
        plt.show()

        # Visualize first selected feature from B
        imb = Bp[0, :]
        if np.sum(imb) < 0:
            imb = -imb
        imb = imb.reshape(32, 32)
        
        print(f"imb: {imb.shape}     (Expected: 32x32)")

        plt.figure()
        plt.imshow(imb, aspect='equal', cmap='viridis')
        plt.colorbar()
        plt.title("Bp feature 1")
        plt.show()
        return logits, include_decision, score_map, ratios

# ==========================================
# TEST SCRIPT
# ==========================================
if __name__ == "__main__":
    print("Initializing model...")
    model = Model(in_channels=3, n_classes=19, gsvd_k=50, threshold=0.5)  # ← pass args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    img_path = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/data/cityscapes/leftImg8bit/val/tubingen/tubingen_000000_000019_leftImg8bit.png"      # Image 1 (In Distribution)
    # img_path = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/fishyscapes_rgb_100/0010_04_Maurener_Weg_8_000002_000120_labels.png"       # Image 2 (Out of Distribution)
    # img_path = "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/dataset_AnomalyTrack/images/airplane0001.jpg"
    

    # 1. Load Image
    raw_img = Image.open(img_path).convert("RGB")
    transform = Compose([
                ToImage(),
                Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
                ToDtype(dtype=torch.float32, scale=True),
                Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
    # 2. Preprocess into Tensor
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    print("\nRunning forward pass...")
    with torch.no_grad():
        logits, include_decision, score_map, ratios = model(img_tensor)

    print("\n--- Output Shapes ---")
    print(f"Logits shape:    {logits.shape}")
    print(f"Score map shape: {score_map.shape}")
    print(f"Include:         {include_decision}")        # bool, no .shape
    print(f"Top ratios:      {ratios[:10].tolist()}")