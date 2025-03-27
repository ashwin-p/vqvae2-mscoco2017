import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')  # Adjust path as needed

import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from utils.datasets import get_dataloader_mscoco
from tqdm import tqdm
import numpy as np

def calculate_psnr(img1, img2):
    """ Compute the Peak Signal-to-Noise Ratio (PSNR) between two images. """
    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')  # Perfect reconstruction
    max_pixel = 1.0  # Images are normalized between [0,1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item()

def evaluate_psnr():
    # Hyperparameters
    batch_size = 128
    images_dir = '/root/workspace/coco2017/test2017'  # Update as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load trained VQ-VAE-2 model
    model = VQVAE().to(device)
    checkpoint = '/root/workspace/vqvae2-mscoco2017/vqvae_latest.pth'  # Update checkpoint path
    model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])
    model.eval()

    # Prepare DataLoader (No captions needed for evaluation)
    dataloader = get_dataloader_mscoco(
        images_dir,
        captions_file=None,  # No captions required
        batch_size=batch_size,
        clip_model=None,  # No CLIP model needed
        device=device,
        shuffle=False,
        num_workers=16,
        persistent_workers=True,
        distributed=False
    )

    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Evaluating PSNR"):
            images = images.to(device)
            x_recon, _ = model(images)  # Pass images through VQ-VAE-2

            psnr_batch = calculate_psnr(images, x_recon)
            total_psnr += psnr_batch * images.shape[0]
            num_samples += images.shape[0]

    avg_psnr = total_psnr / num_samples
    print(f"Average PSNR: {avg_psnr:.4f} dB")

if __name__ == "__main__":
    evaluate_psnr()

