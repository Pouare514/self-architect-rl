import torch
import torch.nn as nn
from torch.distributions import Categorical
from .graph import ModuleBuilder

class BasicPolicyBlock(nn.Module):
    def __init__(self, latent_dim=128, action_dim=7, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, z_t):
        logits = self.net(z_t)
        return Categorical(logits=logits)

ModuleBuilder.register_layer("basic_policy", BasicPolicyBlock)

class PolicyModule(nn.Module):
    """
    Prend z_t (et potentiellement l'état caché du world model)
    et sort une distribution d'actions.
    """
    def __init__(self, graph_def):
        super().__init__()
        self.net = ModuleBuilder.build(graph_def)
        
    def forward(self, z_t):
        logits = self.net(z_t)
        # Si la sortie est déjà Categorical (ancien basic_policy), la retourner
        if isinstance(logits, Categorical):
            return logits
        return Categorical(logits=logits)
