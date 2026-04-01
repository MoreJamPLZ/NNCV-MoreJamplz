"""
Training script for fine-tuning SegFormer-B5 on Cityscapes.
Uses the Model wrapper class for compatibility with the challenge submission format.

Usage:
    python train.py --data-dir ./data/cityscapes --epochs 20 --lr 6e-5 --batch-size 4
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import Cityscapes
from torchvision.utils import make_grid
from torchvision.transforms.v2 import (
    Compose,
    Normalize,
    Resize,
    ToImage,
    ToDtype,
    InterpolationMode,
)
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import SegformerForSemanticSegmentation

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}

def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color for visualization
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)
    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id
        for i in range(3):
            color_image[:, i][mask] = color[i]
    return color_image


def get_args_parser():
    parser = ArgumentParser("Training script for SegFormer-B5")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to Cityscapes data")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size (SegFormer-B5 is large, use small batch)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=6e-5, help="Learning rate (lower for fine-tuning)")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="segformer-b5-finetune", help="Experiment ID for W&B")
    parser.add_argument("--from-pretrained", action="store_true", default=True,
                        help="Initialize from pretrained HuggingFace weights (default: True)")
    parser.add_argument("--img-size", type=int, default=1024, help="Input image size (1024 matches pretrained)")
    return parser


def main(args):
    wandb.init(
        project="5lsm0-cityscapes-segmentation",
        name=args.experiment_id,
        config=vars(args),
    )

    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SegFormer uses ImageNet normalization
    img_transform = Compose([
        ToImage(),
        Resize((args.img_size, args.img_size), interpolation=InterpolationMode.BILINEAR),
        ToDtype(torch.float32, scale=True),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    target_transform = Compose([
        ToImage(),
        Resize((args.img_size, args.img_size), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),
    ])

    train_dataset = Cityscapes(
        args.data_dir, split="train", mode="fine", target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )
    valid_dataset = Cityscapes(
        args.data_dir, split="val", mode="fine", target_type="semantic",
        transform=img_transform, target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # Initialize model
    model = Model(in_channels=3, n_classes=19)

    if args.from_pretrained:
        print("Loading pretrained nvidia/segformer-b5-finetuned-cityscapes-1024-1024 ...")
        pretrained = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        )
        model.segformer.load_state_dict(pretrained.state_dict())
        del pretrained
        print("Pretrained weights loaded.")

    model = model.to(device)

    # Loss — SegFormer outputs at 1/4 resolution, so downsample labels
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # AdamW with low LR for fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    best_valid_loss = float('inf')
    current_best_model_path = None

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # --- Training ---
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):
            labels = convert_to_train_id(labels)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long().squeeze(1)  # (B, H, W)

            optimizer.zero_grad()
            logits = model(images)  # (B, 19, H/4, W/4)

            # Downsample labels to match logit resolution
            labels_down = nn.functional.interpolate(
                labels.unsqueeze(1).float(),
                size=logits.shape[2:],
                mode="nearest"
            ).squeeze(1).long()

            loss = criterion(logits, labels_down)
            loss.backward()
            optimizer.step()
            scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)

        # --- Validation ---
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):
                labels = convert_to_train_id(labels)
                images, labels = images.to(device), labels.to(device)
                labels = labels.long().squeeze(1)

                logits = model(images)

                labels_down = nn.functional.interpolate(
                    labels.unsqueeze(1).float(),
                    size=logits.shape[2:],
                    mode="nearest"
                ).squeeze(1).long()

                loss = criterion(logits, labels_down)
                losses.append(loss.item())

                if i == 0:
                    # Upsample predictions for visualization
                    preds_up = nn.functional.interpolate(
                        logits, size=labels.shape[1:], mode="bilinear", align_corners=False
                    )
                    predictions = preds_up.argmax(1).unsqueeze(1)
                    labels_vis = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels_vis = convert_train_id_to_color(labels_vis)

                    pred_grid = make_grid(predictions.cpu(), nrow=4).permute(1, 2, 0).numpy()
                    label_grid = make_grid(labels_vis.cpu(), nrow=4).permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(pred_grid)],
                        "labels": [wandb.Image(label_grid)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)

            valid_loss = sum(losses) / len(losses)
            wandb.log({"valid_loss": valid_loss}, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir,
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)
                print(f"  New best model saved: val_loss={valid_loss:.4f}")

    print("Training complete!")

    torch.save(
        model.state_dict(),
        os.path.join(output_dir, f"final_model-epoch={epoch:04}-val_loss={valid_loss:.4f}.pt")
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
