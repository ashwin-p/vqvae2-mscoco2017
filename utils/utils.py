# utils/utils.py

import torch
import os

def save_checkpoint(model, optimizer, epoch, filename):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename, device='cuda'):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        return start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

def save_generated_images(images, epoch, output_dir):
    grid = torchvision.utils.make_grid(images)
    torchvision.utils.save_image(grid, os.path.join(output_dir, f'epoch_{epoch}.png'))
