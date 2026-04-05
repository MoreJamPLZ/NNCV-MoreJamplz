import torch
import torch.nn as nn
from transformers import Dinov2Model, SegformerForSemanticSegmentation, SegformerConfig
import torch.nn.functional as F
from torchvision.transforms.v2 import Compose, ToImage, Resize, ToDtype, Normalize, InterpolationMode
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels

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

        N = B_tokens.shape[1]          
        h = w = int(N ** 0.5)          

        B_grid = B_tokens.permute(0, 2, 1).reshape(1, 768, h, w)

        B_aligned = F.interpolate(
            B_grid,
            size=(A_spatial.shape[2], A_spatial.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        A = A_spatial.squeeze(0).flatten(1).T
        B = B_aligned.squeeze(0).flatten(1).T

        A_norm = F.normalize(A, dim=1)  
        B_norm = F.normalize(B, dim=1)

        scipy.io.savemat('feature_norms.mat', {
                    'A_norm': A_norm.detach().cpu().numpy(),
                    'B_norm': B_norm.detach().cpu().numpy()
                })

        C = A_norm.T @ B_norm / A_norm.shape[0]

        U, S, Vh = torch.linalg.svd(C, full_matrices=False)

        k = 320  # only top 100 directions
        A_proj = A_norm @ U[:, :k]       
        B_proj = B_norm @ Vh[:k, :].T    

        disagreement = (A_proj - B_proj).norm(dim=1)  

        # reshape back to spatial map
        score_map = disagreement.reshape(A_spatial.shape[2], A_spatial.shape[3])   

        image_score = torch.quantile(disagreement, 0.90).item()

        THRESHOLD = 1.15
        include_decision = image_score > THRESHOLD

        print(f"Plane image score:  {image_score:.4f}")
        print(f"Street image score: {image_score:.4f}")

        # FIX: We now return the score_map as well!
        return logits, include_decision, score_map


# ==========================================
# VISUALIZATION UTILITY
# ==========================================
def save_overlay(original_img_pil, score_map_tensor, save_path):
    """
    Takes the 63x63 score map, resizes it to match the original image,
    applies a thermal colormap, blends them, and saves to disk.
    """
    W, H = original_img_pil.size
    
    # 1. Resize the 63x63 score_map back to original image dimensions
    # Unsqueeze to add Batch and Channel dims: (1, 1, 63, 63)
    score_map_resized = F.interpolate(
        score_map_tensor.unsqueeze(0).unsqueeze(0), 
        size=(H, W), 
        mode='bilinear', 
        align_corners=False
    )
    
    # Squeeze back down to 2D numpy array: (H, W)
    score_map_np = score_map_resized.squeeze().cpu().numpy()

    # 2. Normalize scores to [0, 1] so the colormap applies properly
    min_val = score_map_np.min()
    max_val = score_map_np.max()
    score_map_norm = (score_map_np - min_val) / (max_val - min_val + 1e-8)

    # 3. Apply the "Jet" colormap (Blue = low anomaly, Red = high anomaly)
    cmap = plt.get_cmap('jet')
    heatmap = cmap(score_map_norm)[:, :, :3] # Ignore the alpha channel
    
    # 4. Blend the heatmap with the original image (50% opacity)
    img_array = np.array(original_img_pil) / 255.0
    overlay = (0.5 * img_array) + (0.5 * heatmap)

    # 5. Save the image
    plt.imsave(save_path, overlay)
    print(f"Saved heatmap overlay to {save_path}")


# ==========================================
# TEST SCRIPT WITH REAL IMAGES
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Initializing model...")
    model = Model(in_channels=3, n_classes=19).to(device)
    model.eval()

    # Create the preprocessing pipeline
    # IMPORTANT: We use 1008x1008 because it is divisible by both 16 and 14!
    transform = Compose([
        ToImage(),
        Resize(size=(512, 512), interpolation=InterpolationMode.BILINEAR),
        ToDtype(dtype=torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    # --- SETUP YOUR IMAGE PATHS HERE ---
    # Replace these with real paths on your computer
    image_paths = [
        "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/data/cityscapes/leftImg8bit/val/tubingen/tubingen_000000_000019_leftImg8bit.png",       # Image 1 (In Distribution)
        # "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/fishyscapes_rgb_100/0010_04_Maurener_Weg_8_000002_000120_labels.png"       # Image 2 (Out of Distribution)
        # "/Users/mirjamh/Documents/Projects/Neural networks for computer vision/NNCV-MoreJamplz/dataset_AnomalyTrack/images/airplane0001.jpg"
    ]

    for i, img_path in enumerate(image_paths):
        try:
            # 1. Load Image
            raw_img = Image.open(img_path).convert("RGB")
            
            # 2. Preprocess into Tensor
            img_tensor = transform(raw_img).unsqueeze(0).to(device)

            # 3. Run through the model
            print(f"\nProcessing Image {i+1}...")
            with torch.no_grad():
                logits, include_decision, score_map = model(img_tensor)

            # 4. Generate and save the heatmap overlay
            save_name = f"anomaly_overlay_{i+1}.png"
            
            # We pass the raw image so it looks correct, and the raw score_map
            save_overlay(raw_img, score_map, save_name)
            
            print(f"Include Decision (True = Normal, False = OOD): {include_decision}")
            
        except Exception as e:
            print(f"Skipping image {i+1} because of error: {e}")