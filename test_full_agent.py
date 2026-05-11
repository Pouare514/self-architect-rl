import torch
from modules.graph import GraphDef
from modules.perception import PerceptionModule
from modules.world_model import WorldModelModule
from modules.policy import PolicyModule
from modules.value import ValueModule

def test():
    # 1. Init Perception
    g_perc = GraphDef().add_node("jepa", "jepa_encoder", {"in_channels": 3, "img_size": 64}).add_edge("input", "jepa")
    perception = PerceptionModule(g_perc)
    
    # 2. Init World Model
    g_wm = GraphDef().add_node("wm", "basic_world_model", {"latent_dim": 128, "action_dim": 7}).add_edge("input", "wm")
    world_model = WorldModelModule(g_wm)
    
    # 3. Init Policy & Value
    g_pol = GraphDef().add_node("pol", "basic_policy", {"latent_dim": 128, "action_dim": 7}).add_edge("input", "pol")
    policy = PolicyModule(g_pol)
    
    g_val = GraphDef().add_node("val", "basic_value", {"latent_dim": 128}).add_edge("input", "val")
    value = ValueModule(g_val)
    
    # --- Forward Pass Test ---
    B = 2
    dummy_obs = torch.randn(B, 3, 64, 64)
    
    # Perception
    z_t = perception(dummy_obs)
    assert z_t.shape == (B, 128)
    
    # Policy & Value
    action_dist = policy(z_t)
    action = action_dist.sample() # (B,)
    
    # Convert action to one-hot for world model
    action_onehot = torch.nn.functional.one_hot(action, num_classes=7).float()
    
    v_t = value(z_t)
    assert v_t.shape == (B, 1)
    
    # World Model
    wm_out = world_model(z_t, action_onehot, h_prev=None)
    z_next = wm_out['z_next']
    r_pred = wm_out['r_pred']
    h_t = wm_out['h_t']
    
    assert z_next.shape == (B, 128)
    assert r_pred.shape == (B, 1)
    assert h_t.shape == (B, 256)
    
    print("Full agent forward pass: SUCCESS")

if __name__ == "__main__":
    test()
