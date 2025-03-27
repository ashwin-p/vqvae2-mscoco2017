# scripts/generate.py
import sys
sys.path.append('/kaggle/working/code/')
import torch
from models.vqvae import VQVAE
from models.clip_model import CLIPTextEncoder
from models.text2latent import TextToLatentMapper
from PIL import Image
import torchvision.transforms as transforms
from IPython.display import display

def generate_image_from_text(prompt):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained VQ-VAE model
    vqvae = VQVAE().to(device)
    vqvae_checkpoint = '/kaggle/working/vqvae_epoch_best.pth'
    vqvae.load_state_dict(torch.load(vqvae_checkpoint, map_location=device)['state_dict'])
    vqvae.eval()

    # Obtain latent dimension
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 256, 256).to(device)
        quant_t, quant_b, _, _, _ = vqvae.encode(dummy_input)

        quant_t_shape = quant_t.shape
        quant_b_shape = quant_b.shape

        D_t = quant_t.view(quant_t.size(0), -1).size(1)
        D_b = quant_b.view(quant_b.size(0), -1).size(1)

        total_latent_dim = D_t + D_b  

    # Load the trained TextToLatentMapper
    text2latent = TextToLatentMapper(latent_dim=total_latent_dim).to(device)
    text2latent_checkpoint = '/kaggle/working/text2latent_epoch_best.pth'
    text2latent.load_state_dict(torch.load(text2latent_checkpoint, map_location=device)['state_dict'])
    text2latent.eval()

    # Load the CLIP model
    clip_text_encoder = CLIPTextEncoder(device=device)
    clip_text_encoder.eval()

    with torch.no_grad():
        # Encode text prompt
        text_embedding = clip_text_encoder([prompt])
        text_embedding = text_embedding.float()

        # Map text embedding to latent vector
        predicted_latent = text2latent(text_embedding)
        predicted_latent = predicted_latent.float()

        # Split predicted latent vector into quant_t and quant_b
        quant_t_flat = predicted_latent[:, :D_t]
        quant_b_flat = predicted_latent[:, D_t:D_t + D_b]

        # Reshape to match expected input shapes
        quant_t = quant_t_flat.view(quant_t_shape).to(device)
        quant_b = quant_b_flat.view(quant_b_shape).to(device)

        # Decode image
        x_recon = vqvae.decode(quant_t, quant_b)
        x_recon = x_recon.clamp(0, 1)

    # Convert to PIL Image and display
    transform = transforms.ToPILImage()
    img = transform(x_recon.squeeze().cpu())

    # Display the image
    display(img)

    # Optionally save the image
    img.save(prompt+'.png')

if __name__ == "__main__":
    l = sys.argv[1:]
    prompt = ' '.join(l)
    generate_image_from_text(prompt)

