import torch
import torch.nn as nn
from vit import Attention, FeedForward
from skipnorm import SkipNorm


class ResNet(nn.Module):
    def __init__(self, num_layers, dim, hidden_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers

        for i in range(self.num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            ))

    def forward(self, x):
        for L in self.layers:
            x = x + L(x)
        return x

class SNResNet(nn.Module):
    def __init__(self, num_layers, dim, hidden_dim, sn_window):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.num_layers = num_layers
        self.skipnorm = SkipNorm(dim, window=sn_window, depth=self.num_layers)

        for i in range(self.num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            ))

    def forward(self, x):
        self.skipnorm.reset()
        for i, L in enumerate(self.layers):
            x = x + self.skipnorm(L(x), i)
            print(torch.stack(self.skipnorm.skips).shape)
        return x

class SNTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, window):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.skipnorm = SkipNorm(dim, depth, window)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x):
        self.skipnorm.reset()
        for i, (attn, ff) in enumerate(self.layers):
            s = attn(x) + x
            x = x + self.skipnorm(s + ff(s), i)
        return self.norm(x)

