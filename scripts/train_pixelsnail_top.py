# scripts/train_pixelsnail.py
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
from models.pixelsnail import PixelSNAIL
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import os

def train():
    # Hyperparameters
    num_epochs = 100
    patience = 5
    initial_learning_rate = 1e-3
    min_learning_rate = 1e-6
    batch_size = 32
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

    # Determine the shape of the top latent codes
    sample_data = torch.randn(1, 3, 256, 256).to(device)  # Adjust image size if needed
    with torch.no_grad():
        quant_t, quant_b, _, id_t, id_b = vqvae.encode(sample_data)
    height_t, width_t = id_t.shape[1], id_t.shape[2]
    n_embed_t = vqvae.quantize_t.n_embed  # Number of embeddings in top codebook

    # Load PixelSNAIL model for top level
    pixelsnail_t = PixelSNAIL(
        shape=(height_t, width_t),
        n_class=n_embed_t,
        channel=256,
        kernel_size=5,
        n_block=4,
        n_res_block=4,
        res_channel=128,
        attention=True,
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0
    ).to(device)

    optimizer_t = Adam(pixelsnail_t.parameters(), lr=initial_learning_rate)

    # Initialize scheduler
    scheduler_t = CosineAnnealingLR(optimizer_t, T_max=num_epochs, eta_min=min_learning_rate)

    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()
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
            break

        pixelsnail_t.train()
        train_loss = 0
        loop = tqdm(dataloader)

        for step, (images, _) in enumerate(loop):
            images = images.to(device)
            optimizer_t.zero_grad()

            with torch.no_grad():
                # Get latent codes from VQ-VAE model
                _, _, _, id_t, _ = vqvae.encode(images)
                # id_t: [batch_size, height_t, width_t]

            # Train top PixelSNAIL
            inputs_t = id_t
            targets_t = id_t

            logits_t, _ = pixelsnail_t(inputs_t)

            loss_t = F.cross_entropy(
                logits_t.reshape(-1, n_embed_t),
                targets_t.flatten()
            )

            # Backward pass and optimization
            loss_t.backward()
            optimizer_t.step()

            train_loss += loss_t.item()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss_t.item())

            # Free memory explicitly
            del images, logits_t, loss_t
            torch.cuda.empty_cache()

        avg_train_loss = train_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss}")

        # Adjust learning rate
        scheduler_t.step()

        # Validation phase
        pixelsnail_t.eval()
        val_loss = 0
        with torch.no_grad():
            for images, _ in val_dataloader:
                images = images.to(device)
                with torch.no_grad():
                    _, _, _, id_t, _ = vqvae.encode(images)

                # Validate top PixelSNAIL
                inputs_t = id_t
                targets_t = id_t

                logits_t, _ = pixelsnail_t(inputs_t)

                loss_t = F.cross_entropy(
                    logits_t.reshape(-1, n_embed_t),
                    targets_t.flatten()
                )

                val_loss += loss_t.item()

                # Free memory explicitly
                del images, logits_t, loss_t
                torch.cuda.empty_cache()

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss}")

        # Check for improvement and save models
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(pixelsnail_t, optimizer_t, epoch, f'pixelsnail_t_epoch_best.pth')
            os.system(f"echo Saved PixelSNAIL model at epoch {epoch+1} with loss: {avg_val_loss} >> pixel_top_loss_log.txt")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs, stopping training.")
            early_stop = True

    # Cleanup
    del pixelsnail_t, optimizer_t, dataloader, val_dataloader
    torch.cuda.empty_cache()
    print("Training completed.")

if __name__ == "__main__":
    train()
