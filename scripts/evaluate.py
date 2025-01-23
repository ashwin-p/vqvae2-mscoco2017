# scripts/evaluate.py
import sys
sys.path.append('/kaggle/working/code/')

import torch
import torch.nn.functional as F
from models.vqvae import VQVAE
from utils.datasets import get_dataloader_mscoco
from utils.metrics import calculate_inception_score, calculate_fid_score
from tqdm import tqdm
import numpy as np

def evaluate():
    # Hyperparameters
    batch_size = 64
    images_dir = '/kaggle/input/coco-2017-dataset/coco2017/test2017'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained VQ-VAE model
    model = VQVAE().to(device)
    checkpoint = '/kaggle/working/vqvae_epoch_best.pth'
    model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])
    model.eval()

    # Prepare data loader
    dataloader = get_dataloader_mscoco(images_dir, None, batch_size, shuffle=False, num_workers=4, distributed=False)

    # Inception Model Setup
    from torchvision.models.inception import inception_v3
    inception_model = inception_v3(weights="IMAGENET1K_V1", transform_input=False).to(device)
    inception_model.eval()

    # Disable gradients for Inception model
    for param in inception_model.parameters():
        param.requires_grad = False

    # Variables to store statistics
    all_preds = []
    mu_real = 0
    mu_gen = 0
    sigma_real = 0
    sigma_gen = 0
    num_samples = 0

    with torch.no_grad():
        for images in tqdm(dataloader):
            images = images.to(device)
            x_recon, _ = model(images)

            # Resize images for Inception model
            real_images_resized = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
            recon_images_resized = F.interpolate(x_recon, size=(299, 299), mode='bilinear', align_corners=False)

            # Compute Inception features
            real_features = inception_model(real_images_resized)
            recon_features = inception_model(recon_images_resized)

            # Update mean and covariance for real images
            real_features_np = real_features.cpu().numpy()
            mu_real += real_features_np.sum(axis=0)
            sigma_real += real_features_np.T @ real_features_np
            # Update mean and covariance for generated images
            recon_features_np = recon_features.cpu().numpy()
            mu_gen += recon_features_np.sum(axis=0)
            sigma_gen += recon_features_np.T @ recon_features_np

            num_samples += real_features_np.shape[0]

            # For Inception Score, collect softmax outputs of generated images
            softmax_output = F.softmax(recon_features, dim=1)
            all_preds.append(softmax_output.cpu().numpy())

    # Compute final mean and covariance
    mu_real /= num_samples
    mu_gen /= num_samples
    sigma_real /= num_samples
    sigma_gen /= num_samples

    # Compute FID
    from scipy.linalg import sqrtm

    cov_sqrt, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    if np.iscomplexobj(cov_sqrt):
        cov_sqrt = cov_sqrt.real

    fid_score = np.sum((mu_real - mu_gen) ** 2) + np.trace(sigma_real + sigma_gen - 2 * cov_sqrt)

    # Compute Inception Score
    preds = np.concatenate(all_preds, axis=0)
    split_size = preds.shape[0] // 10  # Use 10 splits
    scores = []
    for i in range(10):
        part = preds[i * split_size: (i + 1) * split_size, :]
        py = np.mean(part, axis=0)
        kl_div = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_div = np.mean(np.sum(kl_div, axis=1))
        scores.append(np.exp(kl_div))
    inception_score = np.mean(scores)

    print(f"Inception Score: {inception_score}")
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    evaluate()

