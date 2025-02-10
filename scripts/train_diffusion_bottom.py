# scripts/train_diffusion_bottom.py
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

# ----------------------------------
# Function to generate top latents using the saved top diffusion model.
# It runs a reverse diffusion loop over T_top timesteps.
# ----------------------------------
def generate_top_latents(text_embeddings, top_diffusion_model, T_top, top_mask_token, top_max_seq_len, device):
    """
    Given a batch of text embeddings, generate discrete top latent tokens by 
    running the reverse diffusion process using the saved top diffusion model.
    
    Args:
        text_embeddings: Tensor of shape [B, 512] from the CLIP text encoder.
        top_diffusion_model: A trained DiffusionTransformer for top latents.
        T_top: Total number of diffusion timesteps for the top branch.
        top_mask_token: The integer value of the [MASK] token for top latents.
        top_max_seq_len: The sequence length for top latents (e.g. 32*32 = 1024).
        device: torch.device.
        
    Returns:
        x_top: Tensor of shape [B, top_max_seq_len] of discrete tokens.
    """
    B = text_embeddings.size(0)
    # Initialize with all [MASK] tokens.
    x_top = torch.full((B, top_max_seq_len), fill_value=top_mask_token, device=device, dtype=torch.long)
    for t_val in range(T_top, 0, -1):
        t_tensor = torch.full((B,), t_val, device=device, dtype=torch.long)
        with torch.no_grad():
            logits = top_diffusion_model(x_top, t_tensor, text_embeddings)  # [B, top_max_seq_len, vocab_size]
            # For simplicity, we use argmax sampling.
            x_top = torch.argmax(logits, dim=-1)
    return x_top

def train():
    num_epochs = 100
    warmup_epochs = 5
    patience = 5
    initial_lr = 1e-4
    min_lr = 1e-6
    batch_size = 128
    gradient_accumulation_steps = 4
    T_bottom = 100
    T_top = 100
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

    # ------------------------------
    # Load the saved top diffusion model (for generating top latents)
    # ------------------------------
    top_max_seq_len = 32 * 32
    n_embed_top = 512
    num_tokens_top = n_embed_top + 1
    # Use the same embed_dim and number of heads; you can adjust num_layers_top as needed.
    top_embed_dim = 256
    num_layers_top = 18
    num_heads = 8
    top_mask_token = num_tokens_top - 1

    top_diffusion_model = DiffusionTransformer(num_tokens_top, top_embed_dim, num_layers_top, num_heads,
                                               max_seq_len=top_max_seq_len).to(device)
    # Load the checkpoint for the top diffusion model.
    top_diff_checkpoint = torch.load('diffusion_top_epoch_best.pth', map_location=device)
    top_diffusion_model.load_state_dict(top_diff_checkpoint['state_dict'])
    top_diffusion_model.eval()

    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()

    train_loader = get_dataloader_mscoco(train_images_dir, train_captions_file, batch_size,
                                         clip_text_encoder, device=device, shuffle=True,
                                         num_workers=26, persistent_workers=True)
    val_loader = get_dataloader_mscoco(val_images_dir, val_captions_file, batch_size,
                                       clip_text_encoder, device=device, shuffle=False,
                                       num_workers=26, persistent_workers=True)

    bottom_max_seq_len = 64 * 64
    # We assume the VQ-VAE bottom branch also uses n_embed = 512 codes plus one [MASK].
    n_embed_bottom = 512
    num_tokens_bottom = n_embed_bottom + 1
    bottom_embed_dim = 256
    num_layers_bottom = 18
    bottom_num_heads = 8
    bottom_mask_token = num_tokens_bottom - 1

    bottom_diffusion_model = DiffusionTransformer(num_tokens_bottom, bottom_embed_dim, num_layers_bottom,
                                                  bottom_num_heads, max_seq_len=bottom_max_seq_len).to(device)
    print_model_parameters(bottom_diffusion_model)

    top_gen_proj = nn.Linear(top_embed_dim, 512).to(device)

    optimizer = Adam(list(bottom_diffusion_model.parameters()) + list(top_gen_proj.parameters()), lr=initial_lr)

    ratio = min_lr / initial_lr
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / warmup_epochs
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, num_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
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

        bottom_diffusion_model.train()
        top_gen_proj.train()
        optimizer.zero_grad()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}] Train")

        for step, (images, text_embeddings) in enumerate(loop):
            images = images.to(device)
            text_embeddings = text_embeddings.to(device)  # [B, 512]

            # Obtain bottom latent tokens from VQ-VAE.
            with torch.no_grad():
                # vqvae.encode returns (quant_t, quant_b, diff, id_t, id_b)
                _, _, _, id_t, id_b = vqvae.encode(images)
                # Use bottom latent indices (id_b) for diffusion training.
                x0_bottom = id_b.view(id_b.size(0), -1)  # [B, 4096]
            
            # Generate top latents using the saved top diffusion model.
            x_top_generated = generate_top_latents(text_embeddings, top_diffusion_model,
                                                   T_top, top_mask_token, top_max_seq_len, device)  # [B, 1024]

            # Convert generated top tokens to a continuous conditioning vector:
            # Use the token embedding from the top diffusion model, then average over the sequence.
            top_emb = top_diffusion_model.token_embedding(x_top_generated)  # [B, 1024, top_embed_dim]
            top_emb_avg = top_emb.mean(dim=1)  # [B, top_embed_dim]
            top_condition_generated = top_gen_proj(top_emb_avg)  # [B, 512]

            # Classifier-Free Guidance
            B = x0_bottom.size(0)
            drop_mask = (torch.rand(B, device=device) < cond_drop_prob).float().unsqueeze(1)
            final_condition = (text_embeddings * (1 - drop_mask)) + top_condition_generated

            # Sample a random diffusion timestep for each sample (bottom branch)
            t = torch.randint(1, T_bottom + 1, (B,), device=device)  # [B]

            with autocast():
                # Apply forward diffusion on the bottom latent tokens.
                x_t = forward_diffusion(x0_bottom, t, num_tokens_bottom, bottom_mask_token)
                # Forward pass through the bottom diffusion transformer.
                logits = bottom_diffusion_model(x_t, t, final_condition)  # [B, 4096, num_tokens_bottom]
                loss = F.cross_entropy(logits.view(-1, num_tokens_bottom), x0_bottom.view(-1))
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * gradient_accumulation_steps
            loop.set_postfix(loss=f"{loss.item() * gradient_accumulation_steps:.4f}")

            del images, text_embeddings, x0_bottom, x_t, logits, loss, x_top_generated, top_emb, top_emb_avg, top_condition_generated, final_condition
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}")

        scheduler.step()  # Update learning rate

        # Validation Loop
        bottom_diffusion_model.eval()
        top_gen_proj.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, text_embeddings in val_loader:
                images = images.to(device)
                text_embeddings = text_embeddings.to(device)
                with torch.no_grad():
                    _, _, _, id_t, id_b = vqvae.encode(images)
                    x0_bottom = id_b.view(id_b.size(0), -1)
                    # For validation, you may choose to use the ground-truth top condition.
                    # However, here we also generate top latents.
                    x_top_generated = generate_top_latents(text_embeddings, top_diffusion_model,
                                                           T_top, top_mask_token, top_max_seq_len, device)
                    top_emb = top_diffusion_model.token_embedding(x_top_generated)
                    top_emb_avg = top_emb.mean(dim=1)
                    top_condition_generated = top_gen_proj(top_emb_avg)
                    combined_condition = text_embeddings + top_condition_generated
                B = x0_bottom.size(0)
                t = torch.randint(1, T_bottom + 1, (B,), device=device)
                x_t = forward_diffusion(x0_bottom, t, num_tokens_bottom, bottom_mask_token)
                logits = bottom_diffusion_model(x_t, t, combined_condition)
                loss = F.cross_entropy(logits.view(-1, num_tokens_bottom), x0_bottom.view(-1))
                val_loss += loss.item()
                del images, text_embeddings, x0_bottom, x_t, logits, loss, x_top_generated, top_emb, top_emb_avg, top_condition_generated, combined_condition
                torch.cuda.empty_cache()
                
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(bottom_diffusion_model, optimizer, epoch, f'diffusion_bottom_epoch_best.pth')
            os.system(f"echo Saved diffusion model at epoch {epoch+1} with loss: {avg_val_loss:.4f} >> diffusion_bottom_loss_log.txt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

    del bottom_diffusion_model, optimizer, scheduler, train_loader, val_loader, scaler, top_gen_proj
    torch.cuda.empty_cache()
    print("Diffusion transformer training on bottom latents (conditioned on generated top latents) completed.")

if __name__ == "__main__":
    train()
