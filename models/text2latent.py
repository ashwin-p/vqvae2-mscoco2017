# models/text2latent.py
import torch
import torch.nn as nn

class TextToLatentMapper(nn.Module):
    def __init__(self, text_embedding_dim=512, latent_dim=None, num_layers=6, num_heads=8, ff_dim=1024, dropout=0.1):
        super(TextToLatentMapper, self).__init__()
        
        if latent_dim is None:
            raise ValueError("latent_dim must be provided during initialization.")

        self.latent_dim = latent_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=text_embedding_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim, 
                dropout=dropout,
                batch_first=True
            ), 
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(text_embedding_dim, latent_dim)

    def forward(self, text_embeddings):
        # Expand dimensions to simulate a sequence (B, 1, D)
        text_embeddings = text_embeddings.unsqueeze(1)
        transformed_embeddings = self.transformer(text_embeddings)
        transformed_embeddings = transformed_embeddings.squeeze(1)  # Remove sequence dim
        latent_vectors = self.fc_out(transformed_embeddings)
        return latent_vectors
