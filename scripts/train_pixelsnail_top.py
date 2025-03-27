# scripts/train_pixelsnail.py
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from models.vqvae import VQVAE
from models.clip_model import CLIPTextEncoder
from models.pixelsnail import PixelSNAIL  # Ensure this is the correct path
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import os
import random

# Module to map CLIP embeddings to a spatial feature map.
class CLIPConditioner(torch.nn.Module):
    def __init__(self, clip_dim, out_channels, height, width):
        super(CLIPConditioner, self).__init__()
        self.fc = torch.nn.Linear(clip_dim, out_channels * height * width)
        self.height = height
        self.width = width
        self.out_channels = out_channels

    def forward(self, x):
        # x: (batch, clip_dim) -> ensure it's float32 to match fc weights.
        x = x.float()
        out = self.fc(x)  # (batch, out_channels * height * width)
        out = out.view(x.size(0), self.out_channels, self.height, self.width)
        return out

def train():
    # Hyperparameters
    num_epochs = 100
    patience = 5
    initial_learning_rate = 2e-4  # Lower LR to reduce instability
    batch_size = 128
    condition_dropout_prob = 0.1  # Drop condition 10% of the time

    # Dataset paths
    images_dir = '/root/workspace/coco2017/train2017'
    captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_train_captions.csv'
    val_images_dir = '/root/workspace/coco2017/val2017'
    val_captions_file = '/root/workspace/vqvae2-mscoco2017/mscoco_val_captions.csv'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize and load the CLIP text encoder.
    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()

    # Load the pretrained VQ-VAE (assumed to be trained already) and freeze its parameters.
    vqvae = VQVAE().to(device)
    vqvae_ckpt = '/root/workspace/vqvae2-mscoco2017/vqvae2_mscoco_best.pth'
    if os.path.exists(vqvae_ckpt):
        print(f"Loading VQVAE checkpoint from {vqvae_ckpt}...")
        ckpt = torch.load(vqvae_ckpt, map_location=device)
        vqvae.load_state_dict(ckpt["state_dict"])
    else:
        print("VQVAE checkpoint not found, exiting.")
        return
    vqvae.eval()
    for param in vqvae.parameters():
        param.requires_grad = False

    # Determine the shape of the top latent codes by passing a dummy image.
    dummy = torch.zeros(1, 3, 256, 256).to(device)
    with torch.no_grad():
        _, _, _, id_t, _ = vqvae.encode(dummy)
    # id_t has shape (batch, H, W) for top latents.
    top_latent_shape = id_t.shape[1:]  # e.g., (H, W)
    print("Top latent shape:", top_latent_shape)

    # n_class is the number of discrete tokens (from VQ-VAE quantizer).
    n_class = vqvae.quantize_t.n_embed

    # Initialize the PixelSNAIL model.
    pixelsnail = PixelSNAIL(
        shape=top_latent_shape,
        n_class=n_class,
        channel=128,
        kernel_size=3,
        n_block=2,
        n_res_block=2,
        res_channel=64,
        attention=True,
        dropout=0.1,
        n_cond_res_block=2,
        cond_res_channel=64,
        cond_res_kernel=3,
        n_out_res_block=1
    ).to(device)

    # Initialize the CLIPConditioner to map CLIP embeddings (dim 512) into a feature map.
    # Here, we no longer multiply by 2 so that the dimensions match the model's expectations.
    clip_conditioner = CLIPConditioner(
        clip_dim=512, 
        out_channels=64,
        height=top_latent_shape[0],  # no multiplication by 2
        width=top_latent_shape[1]    # no multiplication by 2
    ).to(device)

    # Initialize optimizer and scheduler (ReduceLROnPlateau)
    params = list(pixelsnail.parameters()) + list(clip_conditioner.parameters())
    optimizer = Adam(params, lr=initial_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience, verbose=True, min_lr=1e-6)

    # Create GradScaler for mixed precision training.
    scaler = GradScaler()

    # Load datasets.
    dataloader = get_dataloader_mscoco(
        images_dir, captions_file, batch_size, clip_text_encoder,
        device=device, shuffle=True, num_workers=32, persistent_workers=True
    )
    val_dataloader = get_dataloader_mscoco(
        val_images_dir, val_captions_file, batch_size, clip_text_encoder,
        device=device, shuffle=False, num_workers=32, persistent_workers=True
    )

    best_val_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            print("Early stopping triggered.")
            break

        pixelsnail.train()
        clip_conditioner.train()
        train_loss = 0
        loop = tqdm(dataloader)

        for step, (images, text_prompts) in enumerate(loop):
            images = images.to(device)
            text_prompts = text_prompts.to(device)

            optimizer.zero_grad()

            # Encode images using pretrained VQ-VAE to obtain top latent codes.
            with torch.no_grad():
                _, _, _, id_t, _ = vqvae.encode(images)
            target = id_t  # (batch, H, W)

            # With probability 10%, drop conditioning to learn an unconditional prior.
            if random.random() < condition_dropout_prob:
                condition_feature = None
            else:
                # Ensure text_prompts are float32 and pass through the conditioner.
                condition_feature = clip_conditioner(text_prompts.float())

            # Mixed precision forward/backward pass with gradient checkpointing.
            with autocast():
                # Use gradient checkpointing only if condition is provided.
                if condition_feature is not None:
                    # Note: checkpoint requires all inputs to be tensors.
                    output, _ = checkpoint(pixelsnail, target, condition_feature)
                else:
                    output, _ = pixelsnail(target, condition=None)
                loss = F.cross_entropy(output, target.long())

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

            del images, text_prompts, target, output, loss
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss}")

        # Validation phase.
        pixelsnail.eval()
        clip_conditioner.eval()
        val_loss = 0
        with torch.no_grad():
            for images, text_prompts in val_dataloader:
                images = images.to(device)
                text_prompts = text_prompts.to(device)
                _, _, _, id_t, _ = vqvae.encode(images)
                target = id_t

                if random.random() < condition_dropout_prob:
                    condition_feature = None
                else:
                    condition_feature = clip_conditioner(text_prompts.float())

                with autocast():
                    if condition_feature is not None:
                        output, _ = checkpoint(pixelsnail, target, condition_feature)
                    else:
                        output, _ = pixelsnail(target, condition=None)
                    loss = F.cross_entropy(output, target.long())
                val_loss += loss.item()

                del images, text_prompts, target, output, loss
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss}")

        # Adjust learning rate using ReduceLROnPlateau.
        scheduler.step(avg_val_loss)

        # Save checkpoint if validation loss improved.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(pixelsnail, optimizer, epoch, f'pixelsnail_epoch_best.pth')
            os.system(f"echo Saved PixelSNAIL model at epoch {epoch+1} with loss: {avg_val_loss} >> pixelsnail_loss_log.txt")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

    del pixelsnail, clip_conditioner, optimizer, scheduler, dataloader, val_dataloader
    torch.cuda.empty_cache()
    print("PixelSNAIL training completed.")

if __name__ == "__main__":
    train()

