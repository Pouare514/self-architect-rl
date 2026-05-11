import torch
import torch.nn as nn
from .graph import ModuleBuilder

class BasicValueBlock(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, z_t):
        return self.net(z_t)

ModuleBuilder.register_layer("basic_value", BasicValueBlock)

class ValueModule(nn.Module):
    """
    Critic qui évalue l'état latent z_t.
    """
    def __init__(self, graph_def):
        super().__init__()
        self.net = ModuleBuilder.build(graph_def)
        
    def forward(self, z_t):
        return self.net(z_t)
