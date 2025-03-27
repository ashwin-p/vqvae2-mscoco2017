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

def print_model_parameters(model):
    print(f"{'Layer':<30}{'Shape':<30}{'Parameters':<15}")
    print("=" * 75)
    
    train_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_params = param.numel()
            train_params += num_params
            print(f"{name:<30}{str(list(param.shape)):<30}{num_params:<15}")

    print("=" * 75)
    print(f"Total Trainable Parameters: {train_params:,}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,}")
    
