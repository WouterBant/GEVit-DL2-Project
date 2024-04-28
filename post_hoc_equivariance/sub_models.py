import torch
import torch.nn as nn


class ScoringModel(nn.Module):
    """ Simple model for score prediction used in PostHocLearnedScoreAggregation """

    def __init__(self, n_dim=64):
        super().__init__()
        self.fc =nn.Sequential(
            nn.Linear(n_dim, 128),
            nn.GELU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class AttentionBlock(nn.Module):
    """ Attention block used in the Transformer class below """
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout=0.0):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.linear = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn(inp_x, inp_x, inp_x)[0]
        x = x + self.linear(self.layer_norm_2(x))
        return x


class Transformer(nn.Module):
    """ Transformer used for equivariant aggegration of latent representations """
    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()

        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, hidden_dim, num_heads, dropout=dropout) for _ in range(num_layers))
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        B, T, n_emb = x.shape

        # we use a cls token for prediction
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        x = x.transpose(0, 1)
        x = self.transformer(x)

        # return the cls vector
        cls = x[0]
        return cls