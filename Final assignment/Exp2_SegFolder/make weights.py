"""
Convert weights/segformer.pt into model.pt ready for submission.
Run: python make_weights.py
"""
import torch
from model import Model

IN_PATH = "weights/segformer.pt"
OUT_PATH = "Final assignment/model.pt"

def main():
    sd = torch.load(IN_PATH, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {f"segformer.{k}": v for k, v in sd.items()}

    model = Model()
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    torch.save(model.state_dict(), OUT_PATH)
    print(f"Saved {OUT_PATH}")

if __name__ == "__main__":
    main()