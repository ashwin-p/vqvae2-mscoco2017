# scripts/train_vqvae.py
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import functional as F
from models.vqvae import VQVAE
from models.clip_model import CLIPTextEncoder
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import os

def train():
    # Hyperparameters
    num_epochs = 100
    patience = 5
    initial_learning_rate = 2e-4  # Lowered to reduce instability
    batch_size = 128
    gradient_accumulation_steps = 4  # Accumulate gradients over 4 steps
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

    # Log model parameters before training starts
    #print("Model Architecture:")
    #print(model)

    # Initialize optimizer, scheduler
    optimizer = Adam(model.parameters(), lr=initial_learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # Load dataset with proper settings
    dataloader = get_dataloader_mscoco(
        images_dir, captions_file, batch_size, clip_text_encoder,
        device=device, shuffle=True, num_workers=8, persistent_workers=True
    )
    val_dataloader = get_dataloader_mscoco(
        val_images_dir, val_captions_file, batch_size, clip_text_encoder,
        device=device, shuffle=False, num_workers=8, persistent_workers=True
    )

    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        model.train()
        train_loss = 0
        loop = tqdm(dataloader)

        for step, (images, _) in enumerate(loop):
            images = images.to(device)
            optimizer.zero_grad()

            # Forward pass
            x_recon, diff = model(images)
            recon_loss = F.mse_loss(x_recon, images)
            loss = recon_loss + 0.25 * diff

            # Backward pass
            loss.backward()

            # Only update the weights every `gradient_accumulation_steps` steps
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            # Free memory explicitly
            del images, x_recon, loss
            torch.cuda.empty_cache()

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
                loss = recon_loss + 0.25 * diff
                val_loss += loss.item()

                # Free memory explicitly
                del images, x_recon, loss
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss}")

        # Check for improvement and save model with epoch info
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(model, optimizer, epoch, f'vqvae_epoch_best.pth')
            os.system(f"echo Saved model at epoch {epoch+1} with loss: {avg_val_loss} >> loss_log.txt")

        else:
            epochs_no_improve += 1

        # Adjust learning rate using cosine annealing
        scheduler.step()

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

    # Cleanup: explicitly delete CUDA tensors & close dataloader workers
    del model, optimizer, scheduler, dataloader, val_dataloader
    torch.cuda.empty_cache()
    print("Training completed.")

if __name__ == "__main__":
    train()

