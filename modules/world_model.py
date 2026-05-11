import torch
import torch.nn as nn
from .graph import ModuleBuilder

class BasicWorldModelBlock(nn.Module):
    """
    Un bloc World Model simple qui utilise un GRU pour la dynamique latente,
    et un MLP pour prédire la récompense.
    """
    def __init__(self, latent_dim=128, action_dim=7, hidden_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        self.action_encoder = nn.Linear(action_dim, hidden_dim)
        self.rnn = nn.GRUCell(latent_dim + hidden_dim, hidden_dim)
        
        self.z_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.r_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        # inputs is a dict: {'z_t': z_t, 'action': action, 'h_prev': h_prev}
        z_t = inputs['z_t']
        action = inputs['action']
        h_prev = inputs.get('h_prev', None)
        
        if h_prev is None:
            h_prev = torch.zeros(z_t.size(0), self.hidden_dim, device=z_t.device)
            
        a_emb = torch.relu(self.action_encoder(action))
        rnn_input = torch.cat([z_t, a_emb], dim=-1)
        
        h_t = self.rnn(rnn_input, h_prev)
        z_next = self.z_predictor(h_t)
        r_pred = self.r_predictor(h_t)
        
        # Return dict to be compatible with potential DAG routing later
        return {'z_next': z_next, 'r_pred': r_pred, 'h_t': h_t}

ModuleBuilder.register_layer("basic_world_model", BasicWorldModelBlock)

class WorldModelModule(nn.Module):
    """
    Prédit z_{t+1} et le reward r_t à partir de z_t et a_t.
    """
    def __init__(self, graph_def):
        super().__init__()
        self.net = ModuleBuilder.build(graph_def)
        
    def forward(self, z_t, action, h_prev=None):
        inputs = {'z_t': z_t, 'action': action, 'h_prev': h_prev}
        # Notre DAGModule actuel passe l'input dict au premier layer si on l'ajuste
        # Pour simplifier on appelle directement le layer de base
        # (Dans la v2 complète on modifiera le graphe pour router les variables)
        for layer in self.net.layers.values():
            if isinstance(layer, BasicWorldModelBlock):
                return layer(inputs)
        raise ValueError("BasicWorldModelBlock non trouvé")
