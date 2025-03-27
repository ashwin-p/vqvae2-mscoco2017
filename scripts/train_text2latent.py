# scripts/train_text2latent.py

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
from models.text2latent import TextToTopLatentMapper, TextToBottomLatentMapper
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import os

def train():
    # Hyperparameters
    num_epochs = 100
    patience = 5
    initial_learning_rate = 1e-4
    min_learning_rate = 1e-6
    batch_size = 64
    gradient_accumulation_steps = 4 
    images_dir = '/root/workspace/coco2017/train2017'
    captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_train_captions.csv'
    val_images_dir = '/root/workspace/coco2017/val2017'
    val_captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_val_captions.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load pre-trained VQ-VAE-2 model
    vqvae = VQVAE().to(device)
    vqvae_checkpoint = torch.load('vqvae_latest.pth', map_location=device)
    vqvae.load_state_dict(vqvae_checkpoint['state_dict'])
    vqvae.eval()

    # Initialize CLIP model
    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()

    # Initialize TextToLatentMappers
    top_mapper = TextToTopLatentMapper().to(device)
    bottom_mapper = TextToBottomLatentMapper().to(device)

    # Initialize optimizers and schedulers
    optimizer_top = Adam(top_mapper.parameters(), lr=initial_learning_rate)
    scheduler_top = CosineAnnealingLR(optimizer_top, T_max=num_epochs, eta_min=min_learning_rate)

    optimizer_bottom = Adam(bottom_mapper.parameters(), lr=initial_learning_rate)
    scheduler_bottom = CosineAnnealingLR(optimizer_bottom, T_max=num_epochs, eta_min=min_learning_rate)

    # Load dataset
    dataloader = get_dataloader_mscoco(
        images_dir, captions_file, batch_size, clip_text_encoder,
        device=device, shuffle=True, num_workers=16, persistent_workers=True
    )
    val_dataloader = get_dataloader_mscoco(
        val_images_dir, val_captions_file, batch_size, clip_text_encoder,
        device=device, shuffle=False, num_workers=16, persistent_workers=True
    )

    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        top_mapper.train()
        bottom_mapper.train()
        train_loss = 0
        loop = tqdm(dataloader)

        for step, (images, text_embeddings) in enumerate(loop):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)

            optimizer_top.zero_grad()
            optimizer_bottom.zero_grad()

            with torch.no_grad():
                # Get latent codes from VQ-VAE model
                _, _, _, id_t, id_b = vqvae.encode(images)
                quant_t, quant_b, _, _, _ = vqvae.encode(images)
                # quant_t: [batch_size, embed_dim, H_t, W_t]
                # quant_b: [batch_size, embed_dim, H_b, W_b]

            # Forward pass through top mapper
            pred_quant_t = top_mapper(text_embeddings)
            loss_top = F.mse_loss(pred_quant_t, quant_t.detach())

            # Forward pass through bottom mapper
            pred_quant_b = bottom_mapper(text_embeddings, pred_quant_t.detach())
            loss_bottom = F.mse_loss(pred_quant_b, quant_b.detach())

            # Total loss
            loss = loss_top + loss_bottom

            # Backward pass and optimization
            loss.backward()
            optimizer_top.step()
            optimizer_bottom.step()

            train_loss += loss.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            # Free memory explicitly
            del images, text_embeddings, loss, loss_top, loss_bottom
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss}")

        # Adjust learning rate using cosine annealing
        scheduler_top.step()
        scheduler_bottom.step()

        # Validation phase
        top_mapper.eval()
        bottom_mapper.eval()
        val_loss = 0
        with torch.no_grad():
            for images, text_embeddings in val_dataloader:
                images = images.to(device)
                text_embeddings = text_embeddings.to(device)

                with torch.no_grad():
                    _, _, _, id_t, id_b = vqvae.encode(images)
                    quant_t, quant_b, _, _, _ = vqvae.encode(images)

                # Forward pass through top mapper
                pred_quant_t = top_mapper(text_embeddings)
                loss_top = F.mse_loss(pred_quant_t, quant_t.detach())

                # Forward pass through bottom mapper
                pred_quant_b = bottom_mapper(text_embeddings, pred_quant_t.detach())
                loss_bottom = F.mse_loss(pred_quant_b, quant_b.detach())

                # Total loss
                loss = loss_top + loss_bottom

                val_loss += loss.item()

                # Free memory explicitly
                del images, text_embeddings, loss, loss_top, loss_bottom
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss}")

        # Check for improvement and save model with epoch info
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(top_mapper, optimizer_top, epoch, f'top_mapper_epoch_best.pth')
            save_checkpoint(bottom_mapper, optimizer_bottom, epoch, f'bottom_mapper_epoch_best.pth')
            os.system(f"echo Saved mappers at epoch {epoch+1} with loss: {avg_val_loss} >> map_loss_log.txt")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

    # Cleanup: explicitly delete CUDA tensors & close dataloader workers
    del top_mapper, bottom_mapper, optimizer_top, optimizer_bottom, scheduler_top, scheduler_bottom, dataloader, val_dataloader
    torch.cuda.empty_cache()
    print("Training completed.")

if __name__ == "__main__":
    train()
