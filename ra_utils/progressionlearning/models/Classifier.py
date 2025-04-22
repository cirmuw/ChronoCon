import torch
import torch.nn as nn

class MergeWithAttention(nn.Module):
    def __init__(self, embed_dim=480, num_heads=4):
        super(MergeWithAttention, self).__init__()
        
        # Multi-head self-attention layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        
        # Optional: You can add a feedforward layer after attention to process the output
        self.fc = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x should be a list of 4 tensors of shape [bzs, dim]
        # Stack the tensors along a new dimension -> shape: [bzs, 4, dim]
        x = torch.stack(x, dim=1)
        
        # Apply multi-head self-attention
        # Note: `key_padding_mask` can be used to mask out certain positions, but it's not needed here
        attn_output, attn_weights = self.attention(x, x, x)
        
        # Optionally: Pass through a feedforward layer
        output = self.fc(attn_output.mean(dim=1))  # Average over the 4 attention outputs

        return output
    

class AttClassifier(nn.Module):
    def __init__(self, encoder, latent_dim: int = 480, num_heads=4, freeze=True):
        super(AttClassifier, self).__init__()
        if freeze:
            self.encoder = self.freeze_encoder(encoder) 
        else:
            self.encoder = encoder
        self.attention = MergeWithAttention(embed_dim=latent_dim, num_heads=num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )
  
    def freeze_encoder(self, encoder):
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder

    def forward(self, timepoints):
        timepoints_embed = []
        for t in timepoints:
            _, latent, _ = self.encoder(t)
            timepoints_embed.append(latent)
        x = self.attention(timepoints_embed)
        x = self.mlp(x)
        return x


