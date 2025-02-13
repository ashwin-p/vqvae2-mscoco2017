# models/diffusion_model.py

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_timestep_embedding(timesteps, embedding_dim):
    """
    Create sinusoidal timestep embeddings.
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb
    )
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb  # [B, embedding_dim]

class CrossAttention(nn.Module):
    """
    Cross-Attention mechanism for conditioning on text embeddings.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (embed_dim // num_heads) ** -0.5

        self.to_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.to_v = nn.Linear(embed_dim, embed_dim, bias=False)

        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, context):
        B, N, C = x.shape
        _, S, _ = context.shape
        H = self.num_heads

        q = self.to_q(x).reshape(B, N, H, C // H)
        k = self.to_k(context).reshape(B, S, H, C // H)
        v = self.to_v(context).reshape(B, S, H, C // H)

        attn_scores = torch.einsum('bnhd,bshd->bnsh', q, k) * self.scale
        attn_probs = attn_scores.softmax(dim=-1)  # [B, N, S, H]

        attn_output = torch.einsum('bnsh,bshd->bnhd', attn_probs, v)
        attn_output = attn_output.reshape(B, N, C)

        return self.to_out(attn_output)

class TransformerEncoderLayerWithCrossAttention(nn.Module):
    """
    Transformer Encoder Layer with Cross-Attention for text conditioning.
    """

    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.cross_attn = CrossAttention(embed_dim, num_heads)
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src, src_mask, src_key_padding_mask, context):
        # Self-attention
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # Cross-attention
        src2 = self.cross_attn(src.transpose(0, 1), context).transpose(0, 1)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # Feed-forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        return src

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        num_tokens,
        embed_dim,
        num_layers,
        num_heads,
        dropout=0.1,
        max_seq_len=256,
        text_embed_dim=512,
    ):
        """
        num_tokens: total number of discrete tokens (VQ codes plus one [MASK] token)
        embed_dim: transformer embedding dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
        max_seq_len: maximum sequence length of latent tokens (from VQ-VAE)
        text_embed_dim: dimension of the CLIP text embeddings
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
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )

        # Project the CLIP text embedding to embed_dim.
        self.text_proj = nn.Sequential(
            nn.Linear(text_embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # Transformer encoder with cross-attention.
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithCrossAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Final projection to predict distribution (logits) over tokens.
        self.out_proj = nn.Linear(embed_dim, num_tokens)

    def forward(self, x_t, t, text_emb, src_mask=None, src_key_padding_mask=None):
        """
        x_t: [B, N] tensor of noisy latent token indices.
        t: [B] tensor of timesteps (integers between 1 and T)
        text_emb: [B, text_embed_dim] tensor (CLIP text embeddings)
        Returns:
            logits: [B, N, num_tokens] predicted logits over discrete latent tokens.
        """
        B, N = x_t.size()
        # Embed tokens and add positional embeddings.
        token_emb = self.token_embedding(x_t)  # [B, N, embed_dim]
        if N <= self.pos_embedding.size(1):
            token_emb = token_emb + self.pos_embedding[:, :N, :]
        else:
            raise ValueError("Sequence length exceeds maximum allowed.")

        # Timestep conditioning.
        t_emb = get_timestep_embedding(t, self.embed_dim)  # [B, embed_dim]
        t_emb = self.timestep_mlp(t_emb).unsqueeze(1)        # [B, 1, embed_dim]
        x = token_emb + t_emb  # [B, N, embed_dim]

        # Prepare text embeddings for cross-attention.
        text_context = self.text_proj(text_emb)  # [B, embed_dim]
        text_context = text_context.unsqueeze(1)  # [B, 1, embed_dim]

        # Transformer expects shape [N, B, embed_dim].
        x = x.transpose(0, 1)  # [N, B, embed_dim]

        # Pass through transformer layers with cross-attention.
        for layer in self.layers:
            x = layer(
                src=x,
                src_mask=src_mask,
                src_key_padding_mask=src_key_padding_mask,
                context=text_context
            )

        x = x.transpose(0, 1)  # [B, N, embed_dim]
        logits = self.out_proj(x)  # [B, N, num_tokens]
        return logits

def forward_diffusion(x0, t, num_tokens, mask_token, T):
    """
    Apply a forward diffusion (mask-and-replace) on discrete latent tokens.
    """
    B, N = x0.size()
    device = x0.device

    # Compute the masking probabilities based on the timestep.
    timesteps = t.float() / T  # normalized timesteps in [0, 1]
    gamma = timesteps.unsqueeze(1)  # [B, 1]
    beta = torch.clip(1.0 - gamma, min=0.0)  # [B, 1]

    # Sample a random value for each token.
    rand = torch.rand(B, N, device=device)
    x_t = x0.clone()

    # With probability gamma, set token to mask_token.
    mask_cond = rand < gamma
    x_t[mask_cond] = mask_token

    # With probability beta, replace with a random token.
    random_cond = rand >= gamma
    random_tokens = torch.randint(0, num_tokens - 1, (B, N), device=device)
    x_t[random_cond] = random_tokens[random_cond]

    return x_t
