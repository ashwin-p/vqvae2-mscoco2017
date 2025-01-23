# scripts/train_text2latent.py
import sys
sys.path.append('/kaggle/working/code/')

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from models.clip_model import CLIPTextEncoder
from models.text2latent import TextToLatentMapper
from models.vqvae import VQVAE
from utils.datasets import get_dataloader_mscoco
from utils.utils import save_checkpoint
from tqdm import tqdm
import numpy as np
import os

def train(rank, world_size):
    # Hyperparameters
    num_epochs = 100  # Updated to 100 epochs
    patience = 5
    initial_learning_rate = 1e-4
    batch_size = 64
    accumulation_steps = 4
    images_dir = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
    captions_file = '/kaggle/working/mscoco_train_captions.csv'
    val_images_dir = '/kaggle/input/coco-2017-dataset/coco2017/val2017'
    val_captions_file = '/kaggle/working/mscoco_val_captions.csv'

    # Set up distributed device
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Initialize process group
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

    # Initialize models
    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()
    vqvae = VQVAE().to(device)
    vqvae_checkpoint = '/kaggle/working/vqvae_epoch_best.pth'
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=map_location)['state_dict'])
    vqvae.eval()

    # Obtain latent dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        quant_t, quant_b, _, _, _ = vqvae.encode(dummy_input)
        total_latent_dim = quant_t.view(quant_t.size(0), -1).size(1) + quant_b.view(quant_b.size(0), -1).size(1)

    if rank == 0:
        print(f'Total latent dimensions: {total_latent_dim}')

    # Initialize transformer-based TextToLatentMapper
    text2latent = TextToLatentMapper(latent_dim=total_latent_dim).to(device)
    text2latent = DDP(text2latent, device_ids=[rank])

    # Log model parameters before training starts
    if rank == 0:
        print("Text2Latent Model Architecture:")
        print(text2latent)

    optimizer = Adam(text2latent.parameters(), lr=initial_learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()  # Enable mixed precision training

    dataloader = get_dataloader_mscoco(images_dir, captions_file, batch_size, clip_text_encoder,
                                       device=device, shuffle=True, num_workers=4, distributed=True)
    val_dataloader = get_dataloader_mscoco(val_images_dir, val_captions_file, batch_size, clip_text_encoder,
                                           device=device, shuffle=False, num_workers=4, distributed=True)

    criterion = torch.nn.MSELoss()
    best_loss = np.inf
    epochs_no_improve = 0
    early_stop = False

    for epoch in range(num_epochs):
        if early_stop:
            if rank == 0:
                print("Early stopping triggered.")
            break

        dataloader.sampler.set_epoch(epoch)
        val_dataloader.sampler.set_epoch(epoch)
        text2latent.train()

        if rank == 0:
            loop = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            loop = enumerate(dataloader)

        optimizer.zero_grad()
        train_loss = 0

        for batch_idx, (images, captions) in loop:
            images = images.to(device)

            # Get CLIP text embeddings
            with torch.no_grad():
                text_embeddings = clip_text_encoder(captions)
            text_embeddings = text_embeddings.float().to(device)

            # Mixed precision forward pass
            with autocast():
                predicted_latent = text2latent(text_embeddings)

                # Get VQ-VAE latent representations
                with torch.no_grad():
                    quant_t, quant_b, _, _, _ = vqvae.encode(images)
                    quant = torch.cat([quant_t.view(quant_t.size(0), -1), quant_b.view(quant_b.size(0), -1)], dim=1)

                loss = criterion(predicted_latent, quant.detach()) / accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accumulation_steps

            if rank == 0:
                loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
                loop.set_postfix(loss=loss.item() * accumulation_steps)

        avg_train_loss = train_loss / len(dataloader)
        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss}")

            # Validation phase
            text2latent.eval()
            val_loss = 0
            with torch.no_grad():
                for images, captions in val_dataloader:
                    images = images.to(device)

                    # Get CLIP text embeddings
                    text_embeddings = clip_text_encoder(captions).float().to(device)

                    with autocast():
                        predicted_latent = text2latent(text_embeddings)

                        quant_t, quant_b, _, _, _ = vqvae.encode(images)
                        quant = torch.cat([quant_t.view(quant_t.size(0), -1), quant_b.view(quant_b.size(0), -1)], dim=1)

                        loss = criterion(predicted_latent, quant.detach())
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Epoch [{epoch+1}/{num_epochs}] Validation Loss: {avg_val_loss}")

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                epochs_no_improve = 0
                save_checkpoint(text2latent.module, optimizer, epoch, f'text2latent_epoch_{epoch}_best.pth')
            else:
                epochs_no_improve += 1

            scheduler.step(avg_val_loss)

            if epochs_no_improve >= patience:
                print(f"No improvement for {patience} epochs, stopping training.")
                early_stop = True

    dist.destroy_process_group()

def main():
    world_size = 2  # Number of GPUs
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12356'
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()

