# models/diffusion_model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    timesteps: Tensor of shape [B]
    Returns: Tensor of shape [B, embedding_dim]
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class DiffusionTransformer(nn.Module):
    def __init__(self, num_tokens, embed_dim, num_layers, num_heads, dropout=0.1, max_seq_len=256):
        """
        num_tokens: total number of discrete tokens (VQ codes plus one [MASK] token)
        embed_dim: transformer embedding dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        max_seq_len: maximum sequence length of latent tokens (from VQ-VAE)
        """
        super(DiffusionTransformer, self).__init__()
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        # Embedding for discrete latent tokens.
        self.token_embedding = nn.Embedding(num_tokens, embed_dim)
        # Learnable positional embedding.
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))

        # Timestep MLP: maps sinusoidal embedding to transformer dimension.
        self.timestep_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Project the CLIP text embedding (assumed to be 512-d) to embed_dim.
        self.text_proj = nn.Linear(512, embed_dim)

        # Transformer encoder.
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection to predict distribution (logits) over tokens.
        self.out_proj = nn.Linear(embed_dim, num_tokens)

    def forward(self, x_t, t, text_emb):
        """
        x_t: [B, N] tensor of noisy latent token indices.
        t: [B] tensor of timesteps (integers between 1 and T)
        text_emb: [B, 512] tensor (averaged CLIP text embedding for the caption)
        Returns:
            logits: [B, N, num_tokens] predicted logits over discrete latent tokens.
        """
        B, N = x_t.size()
        # Embed tokens and add positional embeddings.
        token_emb = self.token_embedding(x_t)  # [B, N, embed_dim]
        if N <= self.pos_embedding.size(1):
            token_emb = token_emb + self.pos_embedding[:, :N, :]
        # Timestep conditioning.
        t_emb = get_timestep_embedding(t, self.embed_dim)  # [B, embed_dim]
        t_emb = self.timestep_mlp(t_emb).unsqueeze(1)        # [B, 1, embed_dim]
        x = token_emb + t_emb
        # Text conditioning.
        text_cond = self.text_proj(text_emb).unsqueeze(1)    # [B, 1, embed_dim]
        x = x + text_cond  # (Alternatively, you could use cross-attention here.)
        # Transformer expects shape [N, B, embed_dim].
        x = x.transpose(0, 1)
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [B, N, embed_dim]
        logits = self.out_proj(x)  # [B, N, num_tokens]
        return logits

def forward_diffusion(x0, t, num_tokens, mask_token):
    """
    Apply a forward diffusion (mask-and-replace) on discrete latent tokens.
    
    x0: [B, N] ground-truth latent tokens (from VQ-VAE encoding; values in 0..num_tokens-2)
    t: [B] tensor of timesteps (integers in [1, T])
    num_tokens: vocabulary size (VQ codes + 1 for [MASK])
    mask_token: index for the [MASK] token (typically num_tokens - 1)
    
    For a cosine beta schedule:
      - Compute r = t_mean / T, where t_mean is the average timestep of the batch.
      - Then, gamma = 0.9 * sin(r * (pi/2)) is the probability to replace with [MASK],
      - And beta  = 0.1 * sin(r * (pi/2)) is the probability to replace with a random token.
    """
    B, N = x0.size()
    device = x0.device
    T = 100  # total number of diffusion steps
    t_mean = t.float().mean().item()
    r = t_mean / T  # normalized timestep in [0, 1]
    gamma = 0.9 * math.sin(r * (math.pi / 2))
    beta  = 0.1 * math.sin(r * (math.pi / 2))

    # Sample a random value for each token.
    rand = torch.rand(B, N, device=device)
    x_t = x0.clone()
    # With probability gamma, set token to mask_token.
    mask_cond = rand < gamma
    # With probability beta, replace with a random token (sampled uniformly from 0..num_tokens-2).
    random_cond = (rand >= gamma) & (rand < gamma + beta)
    x_t[mask_cond] = mask_token
    random_tokens = torch.randint(0, num_tokens - 1, (B, N), device=device)
    x_t[random_cond] = random_tokens[random_cond]
    return x_t

