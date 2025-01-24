# utils/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd
import clip

from models.clip_model import CLIPTextEncoder


class MSCOCODataset(Dataset):
    def __init__(self, images_dir, captions_file=None, clip_model=None, device='cuda', transform=None, fraction=1.0):
        """
        Args:
            images_dir (str): Directory containing images.
            captions_file (str or None): CSV file containing image filenames and captions (None for evaluation).
            clip_model (CLIPTextEncoder or None): Preloaded CLIP text encoder model (None if captions not used).
            device (str): Device for CLIP model ('cuda' or 'cpu').
            transform (callable, optional): Transformations for images.
            fraction (float, optional): Fraction of the dataset to use (0 < fraction <= 1).
        """
        self.images_dir = images_dir
        self.transform = transform
        self.clip_model = clip_model
        self.device = device

        if captions_file is None:
            # Image-only dataset for evaluation
            self.image_filenames = sorted(os.listdir(images_dir))
            self.image_captions = None  # No captions used
        else:
            # Load captions from CSV
            df = pd.read_csv(captions_file)
            df['image'] = df['image'].astype(str).str.strip()
            df['caption'] = df['caption'].astype(str).str.strip()

            # Build a mapping from image_name to list of captions
            self.image_captions = df.groupby('image')['caption'].apply(list).to_dict()
            self.image_filenames = list(self.image_captions.keys())

        print(f"Using {len(self.image_filenames)} images.")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_filename = self.image_filenames[idx]
        image_path = os.path.join(self.images_dir, image_filename)

        # Open and transform image
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        if self.image_captions is None:
            # Return only image if no captions
            return image

        # Get all captions for the image
        captions = self.image_captions[image_filename]

        # Compute CLIP embeddings for all captions and average them
        with torch.no_grad():
            text_embeddings = self.clip_model(captions)  # Shape: (num_captions, 512)
            avg_text_embedding = text_embeddings.mean(dim=0)  # Shape: (512,)

        return image, avg_text_embedding


def get_dataloader_mscoco(images_dir, captions_file=None, batch_size=64, clip_model=None, device='cuda',
                           shuffle=True, num_workers=2, fraction=1.0, distributed=False, persistent_workers=False):
    """
    Creates a DataLoader for the MS COCO dataset.
    
    Args:
        images_dir (str): Path to image directory.
        captions_file (str or None): Path to captions CSV file, or None for evaluation mode.
        batch_size (int): Batch size.
        clip_model (CLIPTextEncoder or None): CLIP text encoder model (None if evaluation mode).
        device (str): Device to run CLIP model on.
        shuffle (bool): Whether to shuffle dataset.
        num_workers (int): Number of DataLoader workers.
        fraction (float): Fraction of dataset to use.
        distributed (bool): Whether to use DistributedSampler.
        persistent_workers (bool): Whether to keep workers alive for multiple epochs.

    Returns:
        DataLoader: DataLoader for MS COCO dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet normalization
    ])

    dataset = MSCOCODataset(images_dir, captions_file, clip_model, device=device, transform=transform, fraction=fraction)

    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        shuffle = False  # Shuffle is mutually exclusive with DistributedSampler
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,  # Keep workers alive for multiple epochs
        pin_memory=False  # Ensures efficient data transfer to GPU
    )
    return dataloader

