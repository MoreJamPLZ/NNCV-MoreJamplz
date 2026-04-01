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

    def forward(self, x):
        """
        Forward pass.
        Args:
            x: (B, 3, H, W) input tensor
        Returns:
            logits: (B, n_classes, H/4, W/4) — SegFormer outputs at 1/4 resolution
        """
        outputs = self.segformer(pixel_values=x)
        return outputs.logits