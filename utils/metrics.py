# utils/metrics.py

import torch
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import numpy as np
from scipy.linalg import sqrtm

def calculate_inception_score(images, device='cuda', batch_size=32, splits=10):
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear')

    def get_pred(x):
        x = up(x)
        with torch.no_grad():
            x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).cpu().numpy()

    preds = np.zeros((len(images), 1000))
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size].to(device)
        preds[i:i+batch_size] = get_pred(batch)

    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits):(k+1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i]
            scores.append(np.sum(pyx * (np.log(pyx + 1e-16) - np.log(py + 1e-16))))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_fid_score(real_images, generated_images, device='cuda', batch_size=32):
    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    up = torch.nn.Upsample(size=(299, 299), mode='bilinear')

    def get_activations(images):
        activations = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            batch = up(batch)
            with torch.no_grad():
                pred = inception_model(batch)
            activations.append(pred.cpu().numpy())
        activations = np.concatenate(activations, axis=0)
        return activations

    real_acts = get_activations(real_images)
    gen_acts = get_activations(generated_images)

    # Calculate mean and covariance statistics
    mu1 = np.mean(real_acts, axis=0)
    sigma1 = np.cov(real_acts, rowvar=False)
    mu2 = np.mean(gen_acts, axis=0)
    sigma2 = np.cov(gen_acts, rowvar=False)

    # Calculate FID
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    # Check and correct imaginary numbers from sqrtm
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

