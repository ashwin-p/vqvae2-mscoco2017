# scripts/train2_vqvae.py
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')

import torch
import os
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from models.vqvae import VQVAE
from models.clip_model import CLIPTextEncoder
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np

def train():
    # Hyperparameters
    num_epochs = 100
    patience = 5
    initial_learning_rate = 1e-4  # Updated initial LR
    beta = 0.25  # Commitment loss weight
    batch_size = 256
    images_dir = '/root/workspace/coco2017/train2017'
    captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_train_captions.csv'
    val_images_dir = '/root/workspace/coco2017/val2017'
    val_captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_val_captions.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize CLIP model
    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()

    # Initialize model
    model = VQVAE().to(device)

    # Load last checkpoint if available
    checkpoint_path = "/root/workspace/vqvae2-mscoco2017/vqvae_latest.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
    else:
        print("Starting training from scratch")

    # Initialize optimizer and scheduler using ReduceLROnPlateau
    optimizer = Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, min_lr=1e-6)

    # Load dataset with 16 workers
    dataloader = get_dataloader_mscoco(images_dir, captions_file, batch_size, clip_text_encoder,
                                       device=device, shuffle=True, num_workers=32, persistent_workers=True)
    val_dataloader = get_dataloader_mscoco(val_images_dir, val_captions_file, batch_size, clip_text_encoder,
                                           device=device, shuffle=False, num_workers=32, persistent_workers=True)

    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        model.train()
        train_loss = 0
        loop = tqdm(enumerate(dataloader), total=len(dataloader))

        optimizer.zero_grad()
        for step, (images, _) in loop:
            images = images.to(device)

            # Forward pass
            x_recon, diff = model(images)
            recon_loss = F.mse_loss(x_recon, images)
            loss = recon_loss + beta * diff  # Applying β

            # Backward pass and weight update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_dataloader:
                images = images.to(device)
                x_recon, diff = model(images)
                recon_loss = F.mse_loss(x_recon, images)
                loss = recon_loss + beta * diff  # Applying β
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss}")

        # Check for improvement and save model with epoch info
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, "/root/workspace/vqvae2-mscoco2017/vqvae2_mscoco_best.pth")
            os.system(f"echo Saved model at epoch {epoch+1} with loss: {avg_val_loss} >> loss_log2.txt")
        else:
            epochs_no_improve += 1

        # Adjust learning rate using ReduceLROnPlateau scheduler
        scheduler.step(avg_val_loss)

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

if __name__ == "__main__":
    train()
