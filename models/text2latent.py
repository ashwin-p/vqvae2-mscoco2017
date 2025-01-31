# models/text2latent.py

import torch
import torch.nn as nn

class TextToTopLatentMapper(nn.Module):
    def __init__(self, text_embedding_dim=512, latent_dim=64, latent_height=32, latent_width=32, num_layers=6, num_heads=8, ff_dim=1024, dropout=0.1):
        super(TextToTopLatentMapper, self).__init__()
        self.latent_dim = latent_dim
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.sequence_length = latent_height * latent_width
        self.text_embedding_dim = text_embedding_dim  # Added for consistency

        self.positional_embeddings = nn.Parameter(torch.randn(self.sequence_length, text_embedding_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(text_embedding_dim, latent_dim)

    def forward(self, text_embeddings):
        batch_size = text_embeddings.size(0)

        # Expand text embeddings to match sequence length
        text_embeddings = text_embeddings.unsqueeze(1)  # Shape: [batch_size, 1, text_embedding_dim]
        text_embeddings = text_embeddings.repeat(1, self.sequence_length, 1)  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Add positional embeddings
        position_embeddings = self.positional_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, sequence_length, text_embedding_dim]
        transformer_input = text_embeddings + position_embeddings  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Transformer Encoder
        transformed = self.transformer_encoder(transformer_input)  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Map to latent dimensions
        latent_codes = self.fc_out(transformed)  # Shape: [batch_size, sequence_length, latent_dim]

        # Reshape to spatial dimensions
        latent_codes = latent_codes.view(batch_size, self.latent_height, self.latent_width, self.latent_dim)
        latent_codes = latent_codes.permute(0, 3, 1, 2)  # Shape: [batch_size, latent_dim, latent_height, latent_width]

        return latent_codes  # This is quant_t


class TextToBottomLatentMapper(nn.Module):
    def __init__(self, text_embedding_dim=512, latent_dim=64, top_latent_dim=64, top_latent_height=32, top_latent_width=32, latent_height=64, latent_width=64, num_layers=6, num_heads=8, ff_dim=1024, dropout=0.1):
        super(TextToBottomLatentMapper, self).__init__()
        self.latent_dim = latent_dim
        self.latent_height = latent_height
        self.latent_width = latent_width
        self.sequence_length = latent_height * latent_width
        self.text_embedding_dim = text_embedding_dim  # Added this line

        self.positional_embeddings = nn.Parameter(torch.randn(self.sequence_length, text_embedding_dim))

        # Projection of top latent codes to match text embedding dimension
        self.top_latent_proj = nn.Conv2d(top_latent_dim, text_embedding_dim, kernel_size=3, stride=1, padding=1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=text_embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(text_embedding_dim, latent_dim)

    def forward(self, text_embeddings, top_latent_codes):
        batch_size = text_embeddings.size(0)

        # Upsample top_latent_codes to match bottom latent spatial dimensions
        top_latent_codes_upsampled = nn.functional.interpolate(
            top_latent_codes, size=(self.latent_height, self.latent_width),
            mode='bilinear', align_corners=False
        )  # Shape: [batch_size, top_latent_dim, latent_height, latent_width]

        # Project top_latent_codes to text_embedding_dim
        top_latent_proj = self.top_latent_proj(top_latent_codes_upsampled)  # Shape: [batch_size, text_embedding_dim, latent_height, latent_width]
        top_latent_proj = top_latent_proj.permute(0, 2, 3, 1)  # Shape: [batch_size, latent_height, latent_width, text_embedding_dim]
        top_latent_proj = top_latent_proj.view(batch_size, self.sequence_length, self.text_embedding_dim)  # Fixed variable

        # Expand text embeddings
        text_embeddings = text_embeddings.unsqueeze(1).repeat(1, self.sequence_length, 1)  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Add positional embeddings
        position_embeddings = self.positional_embeddings.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Combine text embeddings, top latent projections, and positional embeddings
        transformer_input = text_embeddings + top_latent_proj + position_embeddings  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Transformer Encoder
        transformed = self.transformer_encoder(transformer_input)  # Shape: [batch_size, sequence_length, text_embedding_dim]

        # Map to latent dimensions
        latent_codes = self.fc_out(transformed)  # Shape: [batch_size, sequence_length, latent_dim]

        # Reshape to spatial dimensions
        latent_codes = latent_codes.view(batch_size, self.latent_height, self.latent_width, self.latent_dim)
        latent_codes = latent_codes.permute(0, 3, 1, 2)  # Shape: [batch_size, latent_dim, latent_height, latent_width]

        return latent_codes  # This is quant_b
