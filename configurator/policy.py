import torch
import torch.nn as nn
from torch.distributions import Categorical

class ConfiguratorPolicy(nn.Module):
    """
    Prend en entrée l'état du configurateur c_t (stats, gradients, compute)
    et produit des actions architecturales.
    L'espace d'état (state_dim) : [mean_reward, loss_pol, loss_val, performance_gain, num_params, is_prune_phase]
    L'espace d'action : 
      0: NO_OP
      1: POLICY_ADD_LAYER
      2: VALUE_ADD_LAYER
      3: POLICY_WIDEN
      4: VALUE_WIDEN
      5: POLICY_PRUNE_LAYER
      6: VALUE_PRUNE_LAYER
      7: POLICY_ADD_SKIP
      8: VALUE_ADD_SKIP
    """
    def __init__(self, state_dim=6, action_dim=9, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, c_t):
        logits = self.net(c_t)
        return Categorical(logits=logits)
