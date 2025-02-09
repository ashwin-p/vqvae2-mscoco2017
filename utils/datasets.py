# utils/datasets.py
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

from models.clip_model import CLIPTextEncoder

class MSCOCODataset(Dataset):
    def __init__(self, images_dir, captions_file=None, clip_model=None, device='cuda', transform=None):
        """
        Args:
            images_dir (str): Directory containing images.
            captions_file (str, optional): CSV file containing image filenames and captions.
            clip_model (CLIPTextEncoder, optional): Preloaded CLIP text encoder model.
            device (str): Device for CLIP model ('cuda' or 'cpu').
            transform (callable, optional): Transformations for images.
        """
        self.images_dir = images_dir
        self.transform = transform
        self.clip_model = clip_model  # Ensure the CLIP model is assigned
        self.device = device

        # Load captions if provided
        if captions_file is not None:
            df = pd.read_csv(captions_file)
            df['image'] = df['image'].astype(str).str.strip()
            df['caption'] = df['caption'].astype(str).str.strip()

            # Build a mapping from image_name to list of captions
            self.image_captions = df.groupby('image')['caption'].apply(list).to_dict()
            self.image_filenames = list(self.image_captions.keys())
        else:
            # If no captions, just load images
            self.image_filenames = sorted(os.listdir(images_dir))
            self.image_captions = None
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

        if self.image_captions is not None:
            captions = self.image_captions[image_filename]
            if self.clip_model is None:
                raise ValueError("CLIP model is required when captions are used.")
            # Randomly select one caption every time __getitem__ is called.
            random_caption = random.choice(captions)
            with torch.no_grad():
                text_embeddings = self.clip_model([random_caption]).to(self.device)  # Shape: (1, 512)
                text_embedding = text_embeddings.squeeze(0)  # Shape: (512,)
        else:
            text_embedding = torch.zeros(512)

        return image, text_embedding

def get_dataloader_mscoco(images_dir, captions_file=None, batch_size=64, clip_model=None, device='cuda',
                           shuffle=True, num_workers=4, persistent_workers=False):
    """
    Creates a DataLoader for the MS COCO dataset.
    This version always loads the entire dataset.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = MSCOCODataset(images_dir, captions_file, clip_model, device=device, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers
    )
    return dataloader
