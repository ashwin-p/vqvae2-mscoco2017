# scripts/generate_pixelsnail.py
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')

import torch
import numpy as np
from models.vqvae import VQVAE
from models.pixelsnail import PixelSNAIL
from PIL import Image
import torchvision.transforms as transforms
from IPython.display import display

def generate_images(num_samples=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VQ-VAE-2 Model
    vqvae = VQVAE().to(device)
    vqvae_checkpoint = "/root/workspace/vqvae2-mscoco2017/vqvae_epoch_best.pth"
    vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device)['state_dict'])
    vqvae.eval()

    # Load PixelSNAIL Model
    pixelsnail = PixelSNAIL(
        shape=(64, 32, 32), n_class=512, channel=256, kernel_size=5,
        n_block=6, n_res_block=3, res_channel=128, attention=True
    ).to(device)

    pixelsnail_checkpoint = "/root/workspace/vqvae2-mscoco2017/pixelsnail_epoch_best.pth"
    pixelsnail.load_state_dict(torch.load(pixelsnail_checkpoint, map_location=device)['state_dict'])
    pixelsnail.eval()

    # Generate latents with PixelSNAIL
    with torch.no_grad():
        top_shape = (num_samples, 64, 32, 32)
        quant_t = torch.randint(0, 512, top_shape, dtype=torch.long).to(device)
        logits, _ = pixelsnail(quant_t)
        sampled_quant_t = torch.argmax(logits, dim=1)

        bottom_shape = (num_samples, 64, 64, 64)
        quant_b = torch.randint(0, 512, bottom_shape, dtype=torch.long).to(device)

        # Decode using VQ-VAE-2
        generated_images = vqvae.decode_code(sampled_quant_t, quant_b).clamp(0, 1)

    # Convert to PIL Image and Display
    transform = transforms.ToPILImage()
    for i in range(num_samples):
        img = transform(generated_images[i].cpu())
        display(img)
        img.save(f"generated_image_{i}.png")

if __name__ == "__main__":
    generate_images(num_samples=4)

