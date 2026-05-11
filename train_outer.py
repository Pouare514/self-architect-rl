import argparse
import torch
import torch.nn as nn
from envs.wrappers import make_env
from modules.graph import GraphDef
from modules.perception import PerceptionModule
from modules.world_model import WorldModelModule
from modules.policy import PolicyModule
from modules.value import ValueModule
from configurator.policy import ConfiguratorPolicy
from configurator.graph_modifier import GraphModifier

def init_graphs():
    g_perc = GraphDef().add_node("jepa", "jepa_encoder", {"in_channels": 3, "img_size": 64}).add_edge("input", "jepa")
    g_wm = GraphDef().add_node("wm", "basic_world_model", {"latent_dim": 128, "action_dim": 7}).add_edge("input", "wm")
    
    g_pol = GraphDef()
    g_pol.add_node("pol_lin1", "linear", {"in_features": 128, "out_features": 256}).add_node("pol_relu1", "relu").add_node("pol_out", "linear", {"in_features": 256, "out_features": 7})
    g_pol.add_edge("input", "pol_lin1").add_edge("pol_lin1", "pol_relu1").add_edge("pol_relu1", "pol_out")
    
    g_val = GraphDef()
    g_val.add_node("val_lin1", "linear", {"in_features": 128, "out_features": 256}).add_node("val_relu1", "relu").add_node("val_out", "linear", {"in_features": 256, "out_features": 1})
    g_val.add_edge("input", "val_lin1").add_edge("val_lin1", "val_relu1").add_edge("val_relu1", "val_out")
    
    return {'perception': g_perc, 'world_model': g_wm, 'policy': g_pol, 'value': g_val}

def build_modules(graphs):
    return (
        PerceptionModule(graphs['perception']),
        WorldModelModule(graphs['world_model']),
        PolicyModule(graphs['policy']),
        ValueModule(graphs['value'])
    )

def ppo_inner_loop(env, perception, world_model, policy, value, steps=128, epochs=4, lr=1e-3, gamma=0.99, gae_lambda=0.95, clip_coef=0.2):
    all_params = list(perception.parameters()) + list(world_model.parameters()) + list(policy.parameters()) + list(value.parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    
    obs_b, a_b, logp_b, r_b, v_b, d_b = [], [], [], [], [], []
    
    obs, _ = env.reset()
    ep_rewards = []
    ep_reward = 0.0
    
    for _ in range(steps):
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        with torch.no_grad():
            z_t = perception(obs_tensor)
            action_dist = policy(z_t)
            action = action_dist.sample()
            logp = action_dist.log_prob(action)
            v_t = value(z_t)
        
        next_obs, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        ep_reward += reward
        
        obs_b.append(obs_tensor)
        a_b.append(action)
        logp_b.append(logp)
        r_b.append(reward)
        v_b.append(v_t)
        d_b.append(done)
        
        if done:
            obs, _ = env.reset()
            ep_rewards.append(ep_reward)
            ep_reward = 0.0
        else:
            obs = next_obs
            
    with torch.no_grad():
        obs_tensor = torch.tensor(obs).unsqueeze(0)
        z_t = perception(obs_tensor)
        next_v = value(z_t).squeeze()
        
    advantages = torch.zeros(steps)
    lastgaelam = 0
    for t in reversed(range(steps)):
        if t == steps - 1:
            nextnonterminal = 1.0 - d_b[t]
            nextvalues = next_v
        else:
            nextnonterminal = 1.0 - d_b[t]
            nextvalues = v_b[t+1].squeeze()
            
        delta = r_b[t] + gamma * nextvalues * nextnonterminal - v_b[t].squeeze()
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
        
    returns = advantages + torch.tensor([v.item() for v in v_b])
    
    b_obs = torch.cat(obs_b)
    b_a = torch.cat(a_b)
    b_logp = torch.cat(logp_b)
    b_v = torch.cat(v_b).detach()
    b_returns = returns.unsqueeze(1)
    b_advantages = advantages.unsqueeze(1)
    b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
    
    total_loss_val = 0.0
    for _ in range(epochs):
        z_t = perception(b_obs)
        new_dist = policy(z_t)
        new_logp = new_dist.log_prob(b_a)
        entropy = new_dist.entropy().mean()
        
        logratio = new_logp - b_logp
        ratio = logratio.exp()
        
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        
        new_v = value(z_t)
        
        # Value clipping
        v_loss_unclipped = (new_v - b_returns) ** 2
        v_clipped = b_v + torch.clamp(new_v - b_v, -clip_coef, clip_coef)
        v_loss_clipped = (v_clipped - b_returns) ** 2
        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()
        
        loss = pg_loss - 0.01 * entropy + v_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss_val += loss.item()
        
    mean_rew = sum(ep_rewards)/len(ep_rewards) if ep_rewards else sum(r_b)
    return mean_rew, total_loss_val / epochs

def main():
    parser = argparse.ArgumentParser(description="Outer loop: Entraînement du configurateur d'architecture.")
    parser.add_argument("--env", type=str, default="MiniGrid-DoorKey-8x8-v0")
    parser.add_argument("--meta_iterations", type=int, default=5)
    args = parser.parse_args()
    
    print(f"--- Meta-Training Architect ---")
    env = make_env(args.env)
    
    configurator = ConfiguratorPolicy()
    conf_optimizer = torch.optim.Adam(configurator.parameters(), lr=1e-2)
    modifier = GraphModifier()
    
    graphs = init_graphs()
    perc, wm, pol, val = build_modules(graphs)
    prev_mean_rew = 0.0
    
    for meta_step in range(args.meta_iterations):
        print(f"\n--- Meta Iteration {meta_step+1}/{args.meta_iterations} ---")
        
        mean_rew, mean_loss = ppo_inner_loop(env, perc, wm, pol, val, steps=32)
        num_params = sum(p.numel() for p in pol.parameters()) / 10000.0
        performance_gain = mean_rew - prev_mean_rew
        phase = modifier.set_meta_phase(meta_step)
        print(f"Architect phase: {phase}")
        phase_flag = 1.0 if phase == GraphModifier.PHASE_PRUNE else 0.0
        
        c_t = torch.tensor([mean_rew, mean_loss, mean_loss, performance_gain, num_params, phase_flag], dtype=torch.float32)
        print(f"Configurator State: {['%.4f' % v for v in c_t.tolist()]}")
        
        action_dist = configurator(c_t)
        action = action_dist.sample()
        print(f"Architect Action Selected: {action.item()}")
        
        mutated = modifier.apply_action(graphs, action.item())
        
        if mutated:
            print("Rebuilding modules and transferring weights...")
            new_perc, new_wm, new_pol, new_val = build_modules(graphs)
            
            if action.item() in [1, 3, 5]:
                modifier.transfer_weights(pol, new_pol, graphs['policy'])
            elif action.item() in [2, 4, 6]:
                modifier.transfer_weights(val, new_val, graphs['value'])
                
            perc, wm, pol, val = new_perc, new_wm, new_pol, new_val
        
        # Dynamic penalty on model complexity based on performance improvement
        complexity_penalty = 0.12 * num_params / (1.0 + max(performance_gain, 0.0))
        meta_reward = mean_rew - 0.1 * mean_loss - complexity_penalty
        print(f"Meta reward components: mean_rew={mean_rew:.4f}, mean_loss={mean_loss:.4f}, num_params={num_params:.4f}, complexity_penalty={complexity_penalty:.4f}")
        
        prev_mean_rew = mean_rew
        conf_optimizer.zero_grad()
        meta_loss = -action_dist.log_prob(action) * meta_reward
        meta_loss.backward()
        conf_optimizer.step()
        
    print("\nMéta-entraînement terminé. Architecte fonctionnel !")

if __name__ == "__main__":
    main()
