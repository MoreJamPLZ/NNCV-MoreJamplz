import torch
from transformers import SegformerForSemanticSegmentation

print("1. Downloading pre-trained SegFormer weights from HuggingFace...")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)

# 2. Save only the model weights (state_dict) to a local .pt file
torch.save(model.state_dict(), "segformer.pt")
print("Weights successfully saved to segformer.pt!")