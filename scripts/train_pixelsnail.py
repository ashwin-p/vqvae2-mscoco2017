import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from models.vqvae import VQVAE
from models.pixelsnail import PixelSNAIL
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import os

def train():
    # Hyperparameters
    num_epochs = 100
    patience = 5  # Early stopping patience
    initial_learning_rate = 2e-4
    batch_size = 128
    images_dir = '/root/workspace/coco2017/train2017'
    val_images_dir = '/root/workspace/coco2017/val2017'
    captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_train_captions.csv'
    val_captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_val_captions.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VQ-VAE model
    vqvae = VQVAE().to(device)
    vqvae_checkpoint = '/root/workspace/vqvae2-mscoco2017/vqvae_epoch_best.pth'
    vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device)['state_dict'])
    vqvae.eval()

    # Define PixelSNAIL models
    pixelsnail_top = PixelSNAIL(
        shape=(32, 32), n_class=512, channel=256, kernel_size=5, 
        n_block=8, n_res_block=3, res_channel=128, attention=True
    ).to(device)

    pixelsnail_bottom = PixelSNAIL(
        shape=(64, 64), n_class=512, channel=256, kernel_size=5, 
        n_block=8, n_res_block=3, res_channel=128, attention=True
    ).to(device)

    optimizer_top = Adam(pixelsnail_top.parameters(), lr=initial_learning_rate)
    optimizer_bottom = Adam(pixelsnail_bottom.parameters(), lr=initial_learning_rate)
    scheduler_top = CosineAnnealingLR(optimizer_top, T_max=num_epochs, eta_min=1e-6)
    scheduler_bottom = CosineAnnealingLR(optimizer_bottom, T_max=num_epochs, eta_min=1e-6)

    dataloader = get_dataloader_mscoco(images_dir, captions_file, batch_size, device=device)
    val_dataloader = get_dataloader_mscoco(val_images_dir, val_captions_file, batch_size, device=device)

    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        pixelsnail_top.train()
        pixelsnail_bottom.train()

        train_loss_top = 0
        train_loss_bottom = 0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for images, _ in loop:
            images = images.to(device)

            with torch.no_grad():
                _, _, _, id_t, id_b = vqvae.encode(images)  # Get discrete indices

            optimizer_top.zero_grad()
            optimizer_bottom.zero_grad()

            logits_top = pixelsnail_top(id_t)
            logits_bottom = pixelsnail_bottom(id_b)

            loss_top = F.cross_entropy(logits_top, id_t)
            loss_bottom = F.cross_entropy(logits_bottom, id_b)
            loss = loss_top + loss_bottom

            loss.backward()
            optimizer_top.step()
            optimizer_bottom.step()

            train_loss_top += loss_top.item()
            train_loss_bottom += loss_bottom.item()

            loop.set_postfix(loss_top=loss_top.item(), loss_bottom=loss_bottom.item())

        avg_train_loss_top = train_loss_top / len(dataloader)
        avg_train_loss_bottom = train_loss_bottom / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss (Top): {avg_train_loss_top}, Train Loss (Bottom): {avg_train_loss_bottom}")

        # Validation
        pixelsnail_top.eval()
        pixelsnail_bottom.eval()
        val_loss_top = 0
        val_loss_bottom = 0

        with torch.no_grad():
            for images, _ in val_dataloader:
                images = images.to(device)
                _, _, _, id_t, id_b = vqvae.encode(images)

                logits_top = pixelsnail_top(id_t)
                logits_bottom = pixelsnail_bottom(id_b)

                loss_top = F.cross_entropy(logits_top, id_t)
                loss_bottom = F.cross_entropy(logits_bottom, id_b)

                val_loss_top += loss_top.item()
                val_loss_bottom += loss_bottom.item()

        avg_val_loss_top = val_loss_top / len(val_dataloader)
        avg_val_loss_bottom = val_loss_bottom / len(val_dataloader)
        avg_val_loss = avg_val_loss_top + avg_val_loss_bottom

        print(f"Epoch [{epoch+1}/{num_epochs}] - Val Loss (Top): {avg_val_loss_top}, Val Loss (Bottom): {avg_val_loss_bottom}")

        # Check for improvement and save model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(pixelsnail_top, optimizer_top, epoch, 'pixelsnail_top_best.pth')
            save_checkpoint(pixelsnail_bottom, optimizer_bottom, epoch, 'pixelsnail_bottom_best.pth')
        else:
            epochs_no_improve += 1

        # Update scheduler
        scheduler_top.step()
        scheduler_bottom.step()

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

if __name__ == "__main__":
    train()

