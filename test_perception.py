import torch
from modules.graph import GraphDef
from modules.perception import PerceptionModule

def test():
    # Définition du graphe de perception
    graph = GraphDef()
    # Le jepa_encoder attend in_channels=3, img_size=64
    graph.add_node("jepa", "jepa_encoder", {"in_channels": 3, "patch_size": 8, "embed_dim": 128, "depth": 2, "num_heads": 4, "img_size": 64})
    
    graph.add_edge("input", "jepa")
    
    # Construction du module de perception
    perception = PerceptionModule(graph)
    print(perception)
    
    # Test avec observation dummy de MiniGrid (B, C, H, W)
    dummy_obs = torch.randn(2, 3, 64, 64)
    z_t = perception(dummy_obs)
    
    print(f"Observation shape: {dummy_obs.shape}")
    print(f"Latent z_t shape: {z_t.shape}")
    
    # Le vecteur latent devrait avoir la taille (B, embed_dim) soit (2, 128)
    assert z_t.shape == (2, 128)
    print("Perception module test: SUCCESS")

if __name__ == "__main__":
    test()
