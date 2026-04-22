"""
Convert weights/segformer.pt into model.pt ready for submission.
Just run: python make_model_pt.py
"""
import torch
from model import Model

# edit these
IN_PATH = "weights/segformer.pt"
OUT_PATH = "Final assignment/model.pt"
THRESHOLD = 0.1585  

def main():
    # load the raw HF segformer weights
    sd = torch.load(IN_PATH, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    new_sd = {f"segformer.{k}": v for k, v in sd.items()}
    new_sd["entropy_threshold"] = torch.tensor(float(THRESHOLD))

    # quick sanity check
    model = Model(entropy_threshold=THRESHOLD)
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"missing: {len(missing)}, unexpected: {len(unexpected)}")

    torch.save(model.state_dict(), OUT_PATH)
    print(f"Saved {OUT_PATH} with threshold={THRESHOLD}")


if __name__ == "__main__":
    main()