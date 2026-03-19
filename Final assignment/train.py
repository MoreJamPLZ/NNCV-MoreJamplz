"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
import torchvision.transforms.v2.functional as F
from torch.optim import AdamW
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
    RandomHorizontalFlip,
    ColorJitter
)
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR

from model import Model


# Mapping class IDs to train IDs
id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])

# Mapping train IDs to color
train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels

def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction[:, 0] == train_id

        for i in range(3):
            color_image[:, i][mask] = color[i]

    return color_image


def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch U-Net model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=10, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="unet-training", help="Experiment ID for Weights & Biases")
    parser.add_argument("--exp1", action="store_true", help="Run Experiment 1: Cosine Annealing Scheduler")
    parser.add_argument("--exp1v1", action="store_true", help="Run Experiment 1v1: Cosine Annealing Scheduler without restart")
    parser.add_argument("--exp2", action="store_true", help="Run Experiment 2: Standard Data Augmentation and normalization cityscapes")
    return parser


def main(args):
    # Initialize wandb for logging
    exp_name = args.experiment_id
    tags = []
    
    if args.exp1:
        tags.append("exp1")
    if args.exp1v1:
        tags.append("exp1v1")
    if args.exp2:
        tags.append("exp2")
        
    if tags:
        exp_name = f"{args.experiment_id}-" + "-".join(tags)

    wandb.init(
        project="5lsm0-cityscapes-segmentation",  
        name=exp_name,  
        config=vars(args),  
    )

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    # If you add other sources of randomness (NumPy, Random), 
    # make sure to set their seeds as well
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.exp2:
        # Define the transforms to apply to the data
        img_transform = Compose([
            ToImage(),
            Resize((256, 256)),
            ToDtype(torch.float32, scale=True),
            # Replaced generic (0.5) with Cityscapes mean and std
            Normalize(mean=(0.2869, 0.3251, 0.2839), std=(0.1870, 0.1902, 0.1872)), 
        ])
    else:
        # Define the transforms to apply to the data
        img_transform = Compose([
        ToImage(),
        Resize((256, 256)),
        ToDtype(torch.float32, scale=True),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # Target transform (mask)
    target_transform = Compose([
        ToImage(),
        Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        ToDtype(torch.int64),  # no scaling
    ])

    # Joint transforms for Experiment 2 (applied to both image and mask together)
    def exp2_joint_transforms(image, target):
        # 1. Convert to image and resize both
        image = ToImage()(image)
        target = ToImage()(target)
        
        image = Resize((256, 256))(image)
        target = Resize((256, 256), interpolation=InterpolationMode.NEAREST)(target)
        
        # 2. Joint Spatial Transform (Random Horizontal Flip)
        if torch.rand(1) < 0.5:
            image = F.horizontal_flip(image)
            target = F.horizontal_flip(target)
            
        # 3. Image-only Transform (Color Jitter)
        # We only apply this to the image, so the mask values remain untouched
        image = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(image)
        
        # 4. Final Formatting (dtypes and Cityscapes normalization)
        image = ToDtype(torch.float32, scale=True)(image)
        image = Normalize(mean=(0.2869, 0.3251, 0.2839), std=(0.1870, 0.1902, 0.1872))(image)
        target = ToDtype(torch.int64)(target) # No scaling for mask
        
        return image, target

    # Load the dataset and make a split for training and validation
    if args.exp2:
        # Use the joint transforms for Experiment 2
        train_dataset = Cityscapes(
            args.data_dir,
            split="train",
            mode="fine",
            target_type="semantic",
            transforms=exp2_joint_transforms, 
        )
    else:
        # Use the standard separate transforms for Baseline/Exp1
        train_dataset = Cityscapes(
            args.data_dir,
            split="train",
            mode="fine",
            target_type="semantic",
            transform=img_transform,
            target_transform=target_transform,
        )

    # Validation dataset should not have data augmentation, so we use the standard transforms
    valid_dataset = Cityscapes(
        args.data_dir,
        split="val",
        mode="fine",
        target_type="semantic",
        transform=img_transform,
        target_transform=target_transform,
    )

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = Model(
        in_channels=3,  # RGB images
        n_classes=19,  # 19 classes in the Cityscapes dataset
    ).to(device)

    # Define the loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)  # Ignore the void class

    # Setup Optimizer and Scheduler based on Experiment Flag
    if args.exp1:
        # experiment 1: AdamW + Cosine Annealing Warm Restarts
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    elif args.exp1v1:
        # experiment 1v1: AdamW + Cosine Annealing without restart
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4) # Lower weight decay
        total_steps = args.epochs * len(train_dataloader)
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps) # Stepping per-batch

    else:
        # BASELINE: Standard Adam (or SGD) with a constant learning rate
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None

    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")

        # Training
        model.train()
        for i, (images, labels) in enumerate(train_dataloader):

            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)

            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                if args.exp1:
                    scheduler.step(epoch + i / len(train_dataloader))
                elif args.exp1v1:
                    scheduler.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i)
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            for i, (images, labels) in enumerate(valid_dataloader):

                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)

                labels = labels.long().squeeze(1)  # Remove channel dimension

                outputs = model(images)
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()

                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1)
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
                )
                torch.save(model.state_dict(), current_best_model_path)

    print("Training complete!")

    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pt"
        )
    )
    wandb.finish()


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
