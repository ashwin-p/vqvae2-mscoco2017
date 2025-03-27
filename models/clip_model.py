# models/clip_model.py

import torch
import clip

class CLIPTextEncoder(torch.nn.Module):
    def __init__(self, device='cuda'):
        super(CLIPTextEncoder, self).__init__()
        self.model, _ = clip.load("ViT-B/32", device=device)
        self.device = device

    def forward(self, text):
        text_tokens = clip.tokenize(text, truncate=True).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features

