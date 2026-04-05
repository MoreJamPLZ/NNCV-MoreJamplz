import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


class Model(nn.Module):
    """
    Wrapper around HuggingFace SegFormer-B5 for Cityscapes semantic segmentation.
    
    The class is named 'Model' as required by the challenge server.
    Default constructor args match the pretrained nvidia/segformer-b5-finetuned-cityscapes-1024-1024.
    
    NOTE: SegFormer outputs logits at 1/4 input resolution. 
    Upsampling to original size must happen in postprocess (predict.py).
    """

    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels

        # Hardcoded SegFormer-B5 config — no internet needed at inference
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
        
    # For peak performance, we will load pretrained weights in predict.py and set strict=True to ensure no mismatch. The model class itself is just the architecture definition.
    # def forward(self, x):
    #     """
    #     Forward pass.
    #     Args:
    #         x: (B, 3, H, W) input tensor
    #     Returns:
    #         logits: (B, n_classes, H/4, W/4) — SegFormer outputs at 1/4 resolution
    #     """
    #     outputs = self.segformer(pixel_values=x)
    #     return outputs.logits

    # For the challenge, we need to return both the segmentation logits and an OOD decision. Without threshold
    # def forward(self, x):
    #     outputs = self.segformer(pixel_values=x)
    #     logits = outputs.logits  # segmentation output
        
    #     # OOD decision: for baseline, always predict in-distribution
    #     # True = in-distribution, False = OOD
    #     include_decision = True
        
    #     return logits, include_decision

    def forward(self, x):
        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits  # (B, 19, H/4, W/4)

        # Compute per-image mean entropy as anomaly score
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)  # (B, H/4, W/4)
        mean_entropy = entropy.mean().item()

        # Threshold: images with high entropy are OOD (include=False)
        # Max possible entropy for 19 classes = log(19) ≈ 2.944
        # Start with 50% of max as threshold, tune from results
        THRESHOLD = 0.15  # tune this
        include_decision = mean_entropy < THRESHOLD

        return logits, include_decision
