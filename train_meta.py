"""
train_meta.py — Unified meta-training script.
Combines Phase 4 (multi-task), Phase 5 (distributed rollouts), Phase 6 (WandB logging).

Usage:
    python train_meta.py                          # Single-task, no WandB
    python train_meta.py --multi_task              # Multi-task rotation
    python train_meta.py --multi_task --use_wandb  # Full pipeline with logging
    python train_meta.py --use_ray --num_workers 4 # Distributed rollouts (requires Ray)
"""
import argparse
import torch
import numpy as np

from modules.graph import GraphDef
from modules.perception import PerceptionModule
from modules.world_model import WorldModelModule
from modules.policy import PolicyModule
from modules.value import ValueModule
from configurator.policy import ConfiguratorPolicy
from configurator.graph_modifier import GraphModifier
from envs.wrappers import make_env
from envs.task_sampler import TaskSampler
from utils.logging import MetaTrainingLogger, compute_dag_metrics
from train_outer import ppo_inner_loop, init_graphs, build_modules


def main():
    parser = argparse.ArgumentParser(description="Meta-training: multi-task architect with observability.")
    parser.add_argument("--meta_iterations", type=int, default=20)
    parser.add_argument("--inner_steps", type=int, default=64)
    parser.add_argument("--inner_epochs", type=int, default=4)

    # Phase 4: Multi-task
    parser.add_argument("--multi_task", action="store_true",
                        help="Enable multi-task rotation across MiniGrid environments.")
    parser.add_argument("--task_pool", nargs="+", default=None,
                        help="Custom list of MiniGrid env IDs.")

    # Phase 5: Distributed
    parser.add_argument("--use_ray", action="store_true",
                        help="Use Ray for parallel rollout collection.")
    parser.add_argument("--num_workers", type=int, default=2)

    # Phase 6: Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Enable WandB logging.")
    parser.add_argument("--run_name", type=str, default=None)

    args = parser.parse_args()

    # ── Setup ──────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Self-Architect RL — Meta-Training")
    print("=" * 60)

    # Task sampler (Phase 4)
    if args.multi_task:
        sampler = TaskSampler(task_pool=args.task_pool)
        num_tasks = sampler.num_tasks
        print(f"[Meta] Multi-task mode: {num_tasks} tasks")
        for i in range(num_tasks):
            print(f"  Task {i}: {sampler.get_task_name(i)}")
    else:
        sampler = None
        num_tasks = 1
        single_env = make_env("MiniGrid-DoorKey-8x8-v0")
        print("[Meta] Single-task mode: MiniGrid-DoorKey-8x8-v0")

    # Configurator with task embedding
    # state = [mean_rew, mean_loss, loss_copy, perf_gain, num_params, phase_flag, *task_emb]
    task_emb_dim = num_tasks if args.multi_task else 0
    state_dim = 6 + task_emb_dim

    configurator = ConfiguratorPolicy(state_dim=state_dim, action_dim=9, hidden_dim=64)
    conf_optimizer = torch.optim.Adam(configurator.parameters(), lr=5e-3)
    modifier = GraphModifier()

    graphs = init_graphs()
    perc, wm, pol, val = build_modules(graphs)
    prev_mean_rew = 0.0

    # Logger (Phase 6)
    logger = MetaTrainingLogger(
        project_name="self-architect-rl",
        run_name=args.run_name or ("multi-task" if args.multi_task else "single-task"),
        use_wandb=args.use_wandb,
        config=vars(args),
    )

    # Distributed collector (Phase 5)
    collector = None
    if args.use_ray:
        from distributed.ray_workers import RolloutCollector
        env_ids = (sampler.task_pool if sampler else ["MiniGrid-DoorKey-8x8-v0"])
        collector = RolloutCollector(env_ids, num_workers=args.num_workers)

    # ── Meta-training loop ────────────────────────────────────────────────
    for meta_step in range(args.meta_iterations):
        print(f"\n{'-' * 50}")
        print(f"  Meta Iteration {meta_step + 1}/{args.meta_iterations}")
        print(f"{'-' * 50}")

        # Sample task (Phase 4)
        if sampler:
            task_id = sampler.sample_task_id()
            task_name = sampler.get_task_name(task_id)
            task_emb = sampler.get_task_embedding(task_id)
            env = sampler.get_env(task_id)
        else:
            task_id = 0
            task_name = "MiniGrid-DoorKey-8x8-v0"
            task_emb = np.array([], dtype=np.float32)
            env = single_env

        print(f"  Task: {task_name}")

        # Inner PPO loop
        mean_rew, mean_loss = ppo_inner_loop(
            env, perc, wm, pol, val,
            steps=args.inner_steps,
            epochs=args.inner_epochs,
        )

        # Build configurator state
        num_params = sum(p.numel() for p in pol.parameters()) + \
                     sum(p.numel() for p in val.parameters())
        num_params_scaled = num_params / 10000.0
        performance_gain = mean_rew - prev_mean_rew
        phase = modifier.set_meta_phase(meta_step)

        phase_flag = 1.0 if phase == GraphModifier.PHASE_PRUNE else 0.0
        base_state = [mean_rew, mean_loss, mean_loss, performance_gain,
                       num_params_scaled, phase_flag]
        c_t = torch.tensor(base_state + list(task_emb), dtype=torch.float32)

        # Architect decision
        action_dist = configurator(c_t)
        action = action_dist.sample()
        action_idx = action.item()
        print(f"  Architect: phase={phase}, action={action_idx}")

        # Apply mutation
        mutated = modifier.apply_action(graphs, action_idx)

        if mutated:
            print("  Rebuilding modules and transferring weights...")
            new_perc, new_wm, new_pol, new_val = build_modules(graphs)

            if action_idx in [1, 3, 5, 7]:
                modifier.transfer_weights(pol, new_pol, graphs['policy'])
            elif action_idx in [2, 4, 6, 8]:
                modifier.transfer_weights(val, new_val, graphs['value'])

            perc, wm, pol, val = new_perc, new_wm, new_pol, new_val

        # Meta reward
        complexity_penalty = 0.12 * num_params_scaled / (1.0 + max(performance_gain, 0.0))
        meta_reward = mean_rew - 0.1 * mean_loss - complexity_penalty

        # Update configurator
        conf_optimizer.zero_grad()
        meta_loss = -action_dist.log_prob(action) * meta_reward
        meta_loss.backward()
        conf_optimizer.step()

        prev_mean_rew = mean_rew

        # Log (Phase 6)
        logger.log_meta_step(
            meta_step=meta_step,
            task_name=task_name,
            mean_rew=mean_rew,
            mean_loss=mean_loss,
            num_params=num_params,
            action_taken=action_idx,
            phase=phase,
            graphs=graphs,
            extra_metrics={
                "meta/reward": meta_reward,
                "meta/complexity_penalty": complexity_penalty,
                "meta/performance_gain": performance_gain,
            },
        )

        # Log DAG artifact every 10 steps
        if (meta_step + 1) % 10 == 0:
            logger.log_dag_artifact(graphs, meta_step)

    # ── Cleanup ───────────────────────────────────────────────────────────
    logger.finish()
    if sampler:
        sampler.close_all()
    if collector:
        collector.shutdown()
    if not sampler:
        single_env.close()

    print("\nMeta-training complete.")


if __name__ == "__main__":
    main()
