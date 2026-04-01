import torch
from transformers import SegformerForSemanticSegmentation
from model import Model

print("1. Downloading pre-trained SegFormer weights from HuggingFace...")
pretrained = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
)

print("2. Loading weights into your custom Model wrapper...")
my_model = Model()
my_model.segformer.load_state_dict(pretrained.state_dict())

print("3. Saving to model.pt...")
# This will overwrite your old U-Net model.pt with the SegFormer one
torch.save(my_model.state_dict(), "model.pt")

print("Done! You can now build your Docker image.")