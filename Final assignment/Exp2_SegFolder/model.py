import torch.nn as nn
from transformers import SegformerForSemanticSegmentation, SegformerConfig


def get_segformer_config(n_classes=19):
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


class Model(nn.Module):
    def __init__(self, in_channels=3, n_classes=19):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.segformer = SegformerForSemanticSegmentation(
            get_segformer_config(n_classes)
        )

    def forward(self, x):
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )

        # Raw logits at H/4, W/4
        logits = self.segformer(pixel_values=x).logits

        # Upsample logits back to input size
        logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return logits