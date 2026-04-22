import torch
from model import Model

# Edit if they change
SEGFORMER_PATH = "weights/segformer.pt"
DINO_PATH = "weights/dinov3_vitb16_timm.pth"
NOVICE_PATH = "weights/novice_nonlinear_seed1.pt"
SAVE_PATH = "Final assignment/model.pt"

# From calibration
GSVD_THRESH = 7.5
START_IDX = 256
N_RATIOS = 10


def main():
    print("Loading weights...")

    # Read the files
    seg_weights = torch.load(SEGFORMER_PATH, map_location="cpu")
    dino_weights = torch.load(DINO_PATH, map_location="cpu")
    novice_weights = torch.load(NOVICE_PATH, map_location="cpu")

    # Some checkpoints are wrapped like {"model": ...} or {"state_dict": ...}
    if isinstance(dino_weights, dict) and "model" in dino_weights:
        dino_weights = dino_weights["model"]
    if isinstance(seg_weights, dict) and "state_dict" in seg_weights:
        seg_weights = seg_weights["state_dict"]

    # Create empty dictionary for final combined model
    final_dict = {}

    # Add segformer stuff:
    for k, v in seg_weights.items():
        final_dict["segformer." + k] = v

    # Add DINO stuff:
    for k, v in dino_weights.items():
        final_dict["dino." + k] = v

    # Add Novice stuff:
    for k, v in novice_weights.items():
        final_dict["novice." + k] = v

    # Insert threshold value
    final_dict["gsvd_threshold"] = torch.tensor(float(GSVD_THRESH))

    # Sanity check
    print("Verifying and saving...")

    # Blank model
    my_model = Model(gsvd_threshold=GSVD_THRESH, gsvd_start_idx=START_IDX, gsvd_n_ratios=N_RATIOS)

    # Model with all the weights
    missing, unexpected = my_model.load_state_dict(final_dict, strict=False)
    print(f"Missing: {len(missing)}   Unexpected: {len(unexpected)}")
    if missing:
        print(f"  first 5 missing:    {missing[:5]}")
    if unexpected:
        print(f"  first 5 unexpected: {unexpected[:5]}")

    # Save the model
    torch.save(my_model.state_dict(), SAVE_PATH)
    print(f"Done! Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()