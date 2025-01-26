# scripts/evaluate.py
import sys
sys.path.append('/root/workspace/vqvae2-mscoco2017/')

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.inception import inception_v3
from scipy.linalg import sqrtm
import numpy as np
from tqdm import tqdm
from models.vqvae import VQVAE
from utils.datasets import get_dataloader_mscoco
from utils.metrics import calculate_inception_score

def extract_inception_features(model, images):
    """Extracts features from the penultimate layer of InceptionV3."""
    with torch.no_grad():
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        features = model(images)
    return features

def calculate_fid(real_features, gen_features):
    """Computes the Frechet Inception Distance (FID)."""
    mu_real, sigma_real = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = gen_features.mean(axis=0), np.cov(gen_features, rowvar=False)

    cov_sqrt, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)
    return fid

def evaluate():
    # Hyperparameters
    batch_size = 128
    images_dir = "/root/workspace/coco2017/test2017"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained VQ-VAE-2 model
    model = VQVAE().to(device)
    checkpoint = "/root/workspace/vqvae2-mscoco2017/vqvae_latest.pth"
    model.load_state_dict(torch.load(checkpoint, map_location=device)["state_dict"])
    model.eval()

    # Prepare data loader
    dataloader = get_dataloader_mscoco(images_dir, None, batch_size, shuffle=False, num_workers=16, persistent_workers=True, distributed=False)

    # Load InceptionV3 model (without classification head)
    inception_model = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device)
    inception_model.fc = torch.nn.Identity()  # Extract features before final classification
    inception_model.eval()

    # Disable gradients for efficiency
    for param in inception_model.parameters():
        param.requires_grad = False

    real_features_list, gen_features_list = [], []

    with torch.no_grad():
        for images in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)

            # Generate reconstructed images
            x_recon, _ = model(images)

            # Normalize images before feeding into Inception model
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            real_images = torch.stack([normalize(img) for img in images])
            gen_images = torch.stack([normalize(img) for img in x_recon])
            # Extract features
            real_features = extract_inception_features(inception_model, real_images).cpu().numpy()
            gen_features = extract_inception_features(inception_model, gen_images).cpu().numpy()

            real_features_list.append(real_features)
            gen_features_list.append(gen_features)
        diff = torch.abs(real_images - gen_images)
        mean_diff = diff.mean().item()
        print(f"Mean pixel difference between real and generated images: {mean_diff}")

    # Convert lists to numpy arrays
    real_features = np.concatenate(real_features_list, axis=0)
    gen_features = np.concatenate(gen_features_list, axis=0)

    # Compute FID
    fid_score = calculate_fid(real_features, gen_features)

    # Compute Inception Score
    inception_score = calculate_inception_score(gen_features, device=device)

    print(f"Inception Score: {inception_score}")
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    evaluate()

