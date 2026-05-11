"""
Distributed PPO rollout collection.
Uses Ray for parallel workers if available, otherwise falls back to sequential.
"""
import torch
import torch.nn as nn
import numpy as np

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False


def _collect_single_rollout(env, perception, policy, value, steps):
    """Collect a single rollout (used both by Ray workers and sequential fallback)."""
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

    return {
        "obs": torch.cat(obs_b),
        "actions": torch.cat(a_b),
        "logprobs": torch.cat(logp_b),
        "rewards": r_b,
        "values": [v.squeeze() for v in v_b],
        "dones": d_b,
        "ep_rewards": ep_rewards,
        "last_obs": torch.tensor(obs).unsqueeze(0),
    }


def collect_rollouts_sequential(envs, perception, policy, value, steps_per_env):
    """Collect rollouts sequentially from multiple environments."""
    all_rollouts = []
    for env in envs:
        rollout = _collect_single_rollout(env, perception, policy, value, steps_per_env)
        all_rollouts.append(rollout)
    return all_rollouts


def merge_rollouts(rollouts):
    """Merge multiple rollouts into a single batch."""
    return {
        "obs": torch.cat([r["obs"] for r in rollouts]),
        "actions": torch.cat([r["actions"] for r in rollouts]),
        "logprobs": torch.cat([r["logprobs"] for r in rollouts]),
        "rewards": sum([r["rewards"] for r in rollouts], []),
        "values": sum([r["values"] for r in rollouts], []),
        "dones": sum([r["dones"] for r in rollouts], []),
        "ep_rewards": sum([r["ep_rewards"] for r in rollouts], []),
    }


if HAS_RAY:
    @ray.remote
    class RolloutWorker:
        """
        A Ray remote actor that collects rollouts in a separate process.
        The model weights are synced from the main process before each collection.
        """
        def __init__(self, env_id, make_env_fn_module="envs.wrappers"):
            import importlib
            mod = importlib.import_module(make_env_fn_module)
            self.env = mod.make_env(env_id)

        def collect(self, perception_state, policy_state, value_state,
                    perception_cls, policy_cls, value_cls,
                    graph_configs, steps):
            """
            Rebuild modules from graph configs, load weights, collect rollout.
            graph_configs: dict of serialized graph node/edge data for each module.
            """
            from modules.graph import GraphDef, ModuleBuilder
            from modules.perception import PerceptionModule
            from modules.policy import PolicyModule
            from modules.value import ValueModule

            # Rebuild graphs from serialized configs
            def rebuild_graph(config):
                g = GraphDef()
                for nid, ndata in config["nodes"].items():
                    g.add_node(nid, ndata["layer_type"], ndata.get("config", {}))
                for edata in config["edges"]:
                    g.add_edge(edata["from"], edata["to"],
                               edata.get("type", "sequential"))
                return g

            g_perc = rebuild_graph(graph_configs["perception"])
            g_pol = rebuild_graph(graph_configs["policy"])
            g_val = rebuild_graph(graph_configs["value"])

            perception = PerceptionModule(g_perc)
            perception.load_state_dict(perception_state)
            policy = PolicyModule(g_pol)
            policy.load_state_dict(policy_state)
            value = ValueModule(g_val)
            value.load_state_dict(value_state)

            perception.eval()
            policy.eval()
            value.eval()

            return _collect_single_rollout(self.env, perception, policy, value, steps)


    class DistributedRolloutCollector:
        """Manages a pool of Ray RolloutWorkers."""
        def __init__(self, env_ids, num_workers=None):
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True, log_to_driver=False)

            self.num_workers = num_workers or len(env_ids)
            self.workers = [
                RolloutWorker.remote(env_ids[i % len(env_ids)])
                for i in range(self.num_workers)
            ]
            print(f"[Distributed] Initialized {self.num_workers} Ray workers")

        def collect(self, perception, policy, value, graphs, steps_per_worker):
            """Collect rollouts from all workers in parallel."""
            from utils.logging import serialize_dag_to_json

            perc_state = {k: v.cpu() for k, v in perception.state_dict().items()}
            pol_state = {k: v.cpu() for k, v in policy.state_dict().items()}
            val_state = {k: v.cpu() for k, v in value.state_dict().items()}

            graph_configs = {
                name: serialize_dag_to_json(g) for name, g in graphs.items()
            }

            futures = [
                w.collect.remote(
                    perc_state, pol_state, val_state,
                    type(perception).__name__,
                    type(policy).__name__,
                    type(value).__name__,
                    graph_configs, steps_per_worker
                )
                for w in self.workers
            ]
            rollouts = ray.get(futures)
            return rollouts

        def shutdown(self):
            ray.shutdown()


class RolloutCollector:
    """
    Unified interface: uses Ray if available, otherwise sequential.
    """
    def __init__(self, env_ids, num_workers=2, use_ray=True):
        self.env_ids = env_ids
        self.use_ray = use_ray and HAS_RAY
        self._distributed = None
        self._envs = None

        if self.use_ray:
            self._distributed = DistributedRolloutCollector(env_ids, num_workers)
        else:
            from envs.wrappers import make_env
            self._envs = [make_env(eid) for eid in env_ids]
            print(f"[Distributed] Ray not available, using {len(self._envs)} sequential envs")

    def collect(self, perception, policy, value, graphs, steps_per_env):
        if self.use_ray:
            return self._distributed.collect(
                perception, policy, value, graphs, steps_per_env
            )
        else:
            return collect_rollouts_sequential(
                self._envs, perception, policy, value, steps_per_env
            )

    def shutdown(self):
        if self.use_ray and self._distributed:
            self._distributed.shutdown()
        if self._envs:
            for env in self._envs:
                env.close()
