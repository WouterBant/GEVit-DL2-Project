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
    """ 
    Attention block used in the Transformer class below
    Note that there is no linear layer as this will break equivariance
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(embed_dim)
        self.attn1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.layer_norm_2 = nn.LayerNorm(embed_dim)
        self.act = nn.ReLU()
        self.attn2 = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x):
        inp_x = self.layer_norm_1(x)
        x = x + self.attn1(inp_x, inp_x, inp_x)[0]
        x = self.layer_norm_2(x)
        x = self.act(x) 
        x = x + self.attn2(x, x, x)[0]
        return x


class Transformer(nn.Module):
    """ Transformer used for equivariant aggegration of latent representations """

    def __init__(self, embed_dim, hidden_dim, num_heads, num_layers, dropout=0.0):
        super().__init__()

        self.transformer = nn.Sequential(
            *(AttentionBlock(embed_dim, num_heads) for _ in range(num_layers))
        )
        self.dropout = nn.Dropout(dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x, vis=False):
        B, T, n_emb = x.shape

        # we use a cls token for prediction
        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)

        x = x.transpose(0, 1)
        if vis: 
            inp_x = self.transformer[0].layer_norm_1(x)
            x_attn1 = self.transformer[0].attn1(inp_x, inp_x, inp_x)
            x = x + x_attn1[0]
            x = self.transformer[0].layer_norm_2(x)
            x = self.transformer[0].act(x) 
            x_attn2 = self.transformer[0].attn2(x, x, x)
            x = x + x_attn2[0]

            inp_x = self.transformer[1].layer_norm_1(x)
            x_attn3 = self.transformer[1].attn1(inp_x, inp_x, inp_x)
            x = x + x_attn3[0]
            x = self.transformer[1].layer_norm_2(x)
            x = self.transformer[1].act(x) 
            x_attn4 = self.transformer[1].attn2(x, x, x)

            out = [x_attn1[1], x_attn2[1], x_attn3[1], x_attn4[1]]

            return out
    

        x = self.transformer(x)

        # return the cls vector
        cls = x[0]
        return cls