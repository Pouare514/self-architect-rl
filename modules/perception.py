import torch
import torch.nn as nn
from .graph import ModuleBuilder

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, embed_dim=128, img_size=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        return x

class JEPATransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class JEPAEncoder(nn.Module):
    def __init__(self, in_channels=3, patch_size=8, embed_dim=128, depth=4, num_heads=4, img_size=64):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim, img_size)
        self.blocks = nn.ModuleList([
            JEPATransformerBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x.mean(dim=1) # Global average pooling to get latent vector z_t

class PerceptionModule(nn.Module):
    """
    Encode les observations en un état latent z_t.
    Peut être construit dynamiquement via un GraphDef.
    """
    def __init__(self, graph_def):
        super().__init__()
        self.net = ModuleBuilder.build(graph_def)
        
    def forward(self, obs):
        return self.net(obs)

# Register custom JEPA encoder block
ModuleBuilder.register_layer("jepa_encoder", JEPAEncoder)
