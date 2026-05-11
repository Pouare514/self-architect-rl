import argparse
import torch
from envs.wrappers import make_env
from modules.graph import GraphDef
from modules.perception import PerceptionModule
from modules.world_model import WorldModelModule
from modules.policy import PolicyModule
from modules.value import ValueModule

def init_agent_modules():
    g_perc = GraphDef().add_node("jepa", "jepa_encoder", {"in_channels": 3, "img_size": 64}).add_edge("input", "jepa")
    perception = PerceptionModule(g_perc)
    
    g_wm = GraphDef().add_node("wm", "basic_world_model", {"latent_dim": 128, "action_dim": 7}).add_edge("input", "wm")
    world_model = WorldModelModule(g_wm)
    
    g_pol = GraphDef().add_node("pol", "basic_policy", {"latent_dim": 128, "action_dim": 7}).add_edge("input", "pol")
    policy = PolicyModule(g_pol)
    
    g_val = GraphDef().add_node("val", "basic_value", {"latent_dim": 128}).add_edge("input", "val")
    value = ValueModule(g_val)
    
    return perception, world_model, policy, value

from train_outer import ppo_inner_loop

def main():
    parser = argparse.ArgumentParser(description="Inner loop: Entraînement de l'agent RL avec une architecture fixe.")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0", help="Nom de l'environnement")
    parser.add_argument("--iterations", type=int, default=10, help="Nombre d'itérations PPO")
    args = parser.parse_args()
    
    print(f"Démarrage de l'entraînement PPO sur {args.env} pour {args.iterations} itérations...")
    
    # 1. Initialiser l'environnement
    env = make_env(args.env)
    
    # 2. Construire les modules
    perception, world_model, policy, value = init_agent_modules()
    
    # 3. Boucle RL PPO
    for i in range(args.iterations):
        mean_rew, mean_loss = ppo_inner_loop(env, perception, world_model, policy, value, steps=128, epochs=4)
        print(f"Iteration {i+1}/{args.iterations} - Mean Reward: {mean_rew:.2f} - Mean Loss: {mean_loss:.4f}")
            
    print("Entraînement terminé.")

if __name__ == "__main__":
    main()
