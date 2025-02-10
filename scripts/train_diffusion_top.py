import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from models.vqvae import VQVAE
from models.clip_model import CLIPTextEncoder
from models.diffusion_model import DiffusionTransformer, forward_diffusion
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint, print_model_parameters

def train():
    # Hyperparameters and paths
    num_epochs = 100
    warmup_epochs = 5
    patience = 5
    initial_lr = 1e-4
    min_lr = 1e-6
    batch_size = 128
    gradient_accumulation_steps = 4
    T = 100
    cond_drop_prob = 0.1

    train_images_dir = '/path/to/train/images'
    train_captions_file = '/path/to/train/captions.csv'
    val_images_dir = '/path/to/val/images'
    val_captions_file = '/path/to/val/captions.csv'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vqvae = VQVAE().to(device)
    vqvae_checkpoint = torch.load('vqvae2_mscoco.pth', map_location=device)
    vqvae.load_state_dict(vqvae_checkpoint['state_dict'])
    vqvae.eval()  # Freeze the VQ-VAE

    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()

    train_loader = get_dataloader_mscoco(train_images_dir, train_captions_file, batch_size,
                                         clip_text_encoder, device=device, shuffle=True,
                                         num_workers=26, persistent_workers=True)
    val_loader = get_dataloader_mscoco(val_images_dir, val_captions_file, batch_size,
                                       clip_text_encoder, device=device, shuffle=False,
                                       num_workers=26, persistent_workers=True)

    #The VQ-VAE top branch uses 512 codes; we add one extra token for [MASK].
    n_embed = 512  
    num_tokens = n_embed + 1  # Vocabulary: indices 0..511 for codes; index 512 is [MASK]
    embed_dim = 256
    num_layers = 18
    num_heads = 8
    mask_token = num_tokens - 1

    max_seq_len = 32 * 32

    diffusion_model = DiffusionTransformer(num_tokens, embed_dim, num_layers, num_heads,
                                            max_seq_len=max_seq_len).to(device)
    print_model_parameters(diffusion_model)
    optimizer = Adam(diffusion_model.parameters(), lr=initial_lr)

    # ------------------------------
    # Define a combined warmup + cosine decay scheduler.
    # The scheduler uses a LambdaLR that:
    # - For the first 'warmup_epochs', increases the lr linearly.
    # - Then decays the lr with a cosine schedule from 1.0 to (min_lr/initial_lr).
    # For example, with initial_lr=1e-4 and min_lr=1e-6, the ratio is 0.01.
    # ------------------------------
    ratio = min_lr / initial_lr  # In our case, 1e-6/1e-4 = 0.01
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: at epoch 0: (1/5)=0.2, epoch 1: 0.4, ... epoch 4: 1.0
            return float(epoch + 1) / warmup_epochs
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale the cosine decay to decay from 1.0 to ratio (e.g., 0.01)
            return cosine_decay * (1 - ratio) + ratio

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = GradScaler()  # For mixed precision training

    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    # Training Loop
    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        diffusion_model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")

        for step, (images, text_embeddings) in enumerate(loop):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)  # [B, 512]

            # Obtain top latent tokens from VQ-VAE.
            with torch.no_grad():
                # vqvae.encode returns (quant_t, quant_b, diff, id_t, id_b)
                # Use top latent indices (id_t) for diffusion training.
                _, _, _, id_t, _ = vqvae.encode(images)
                # Flatten spatial dimensions: [B, H, W] -> [B, N]
                x0 = id_t.view(id_t.size(0), -1)

            B = x0.size(0)
            t = torch.randint(1, T + 1, (B,), device=device)  # Random timestep for each sample

            # Classifier-Free Guidance: Drop condition with probability cond_drop_prob.
            drop_mask = (torch.rand(B, device=device) < cond_drop_prob).float().unsqueeze(1)
            # Use a zero vector as the "null" condition.
            text_condition = text_embeddings * (1 - drop_mask)

            with autocast():
                # Apply forward diffusion (mask-and-replace) on the discrete tokens.
                x_t = forward_diffusion(x0, t, num_tokens, mask_token)
                # Forward pass through the diffusion transformer.
                logits = diffusion_model(x_t, t, text_condition)  # [B, N, num_tokens]
                loss = F.cross_entropy(logits.view(-1, num_tokens), x0.view(-1))
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps
            loop.set_postfix(loss=f"{loss.item() * gradient_accumulation_steps:.4f}")

            del images, text_embeddings, x0, x_t, logits, loss
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        scheduler.step()  # Update the learning rate

        # Validation Loop
        diffusion_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, text_embeddings in val_loader:
                images = images.to(device)
                text_embeddings = text_embeddings.to(device)
                with torch.no_grad():
                    _, _, _, id_t, _ = vqvae.encode(images)
                    x0 = id_t.view(id_t.size(0), -1)
                B = x0.size(0)
                t = torch.randint(1, T + 1, (B,), device=device)
                x_t = forward_diffusion(x0, t, num_tokens, mask_token)
                logits = diffusion_model(x_t, t, text_embeddings)
                loss = F.cross_entropy(logits.view(-1, num_tokens), x0.view(-1))
                val_loss += loss.item()
                del images, text_embeddings, x0, x_t, logits, loss
                torch.cuda.empty_cache()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(diffusion_model, optimizer, epoch, f'diffusion_top_epoch_best.pth')
            os.system(f"echo Saved diffusion model at epoch {epoch+1} with loss: {avg_val_loss:.4f} >> diffusion_top_loss_log.txt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

    del diffusion_model, optimizer, scheduler, train_loader, val_loader, scaler
    torch.cuda.empty_cache()
    print("Diffusion transformer training completed.")

if __name__ == "__main__":
    train()
