"""
Observability utilities — DAG metrics extraction and WandB logging.
"""
import json
import time
from collections import defaultdict

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def compute_dag_metrics(graph_def):
    """
    Compute topology metrics from a GraphDef.
    Returns a dict with: num_nodes, num_edges, max_depth, mean_width,
    num_linear, num_skip_adds, total_params_config.
    """
    nodes = graph_def.nodes
    edges = graph_def.edges

    # Build adjacency
    adj = defaultdict(list)
    in_degree = defaultdict(int)
    all_ids = {"input"}
    for e in edges:
        adj[e.from_id].append(e.to_id)
        in_degree[e.to_id] += 1
        all_ids.add(e.from_id)
        all_ids.add(e.to_id)

    # BFS depth
    from collections import deque
    depth = {"input": 0}
    queue = deque(["input"])
    while queue:
        nid = queue.popleft()
        for child in adj[nid]:
            if child not in depth:
                depth[child] = depth[nid] + 1
                queue.append(child)

    max_depth = max(depth.values()) if depth else 0

    # Width per depth level
    width_per_level = defaultdict(int)
    for nid, d in depth.items():
        width_per_level[d] += 1
    mean_width = sum(width_per_level.values()) / max(len(width_per_level), 1)

    # Count node types
    num_linear = sum(1 for n in nodes.values() if n.layer_type == "linear")
    num_skip_adds = sum(1 for n in nodes.values() if n.layer_type == "add")
    num_relu = sum(1 for n in nodes.values() if n.layer_type == "relu")

    # Estimate params from config
    total_params_config = 0
    for n in nodes.values():
        if n.layer_type == "linear":
            inf = n.config.get("in_features", 0)
            outf = n.config.get("out_features", 0)
            total_params_config += inf * outf + outf  # weight + bias

    return {
        "dag/num_nodes": len(nodes),
        "dag/num_edges": len(edges),
        "dag/max_depth": max_depth,
        "dag/mean_width": round(mean_width, 2),
        "dag/num_linear": num_linear,
        "dag/num_relu": num_relu,
        "dag/num_skip_adds": num_skip_adds,
        "dag/total_params_config": total_params_config,
    }


def serialize_dag_to_json(graph_def):
    """Serialize a GraphDef to a JSON-compatible dict for artifact logging."""
    return {
        "nodes": {
            nid: {"layer_type": n.layer_type, "config": n.config}
            for nid, n in graph_def.nodes.items()
        },
        "edges": [
            {"from": e.from_id, "to": e.to_id, "type": e.connection_type}
            for e in graph_def.edges
        ],
    }


class MetaTrainingLogger:
    """
    Unified logger for meta-training. Supports WandB (if available) and console.
    """
    def __init__(self, project_name="self-architect-rl", run_name=None,
                 use_wandb=True, config=None):
        self.use_wandb = use_wandb and HAS_WANDB
        self.step = 0
        self._start_time = time.time()

        if self.use_wandb:
            wandb.init(
                project=project_name,
                name=run_name,
                config=config or {},
                reinit=True,
            )
            print(f"[Logger] WandB initialized: {project_name}/{run_name}")
        else:
            print("[Logger] Running without WandB (console logging only)")

    def log_meta_step(self, meta_step, task_name, mean_rew, mean_loss,
                      num_params, action_taken, phase, graphs,
                      extra_metrics=None):
        """Log all metrics for one meta-training step."""
        self.step = meta_step

        # Compute DAG metrics for each module graph
        all_dag_metrics = {}
        for module_name, graph_def in graphs.items():
            metrics = compute_dag_metrics(graph_def)
            for k, v in metrics.items():
                all_dag_metrics[f"{module_name}/{k}"] = v

        log_dict = {
            "meta_step": meta_step,
            "task": task_name,
            "reward/mean": mean_rew,
            "loss/mean": mean_loss,
            "architect/action": action_taken,
            "architect/phase": 1 if phase == "prune" else 0,
            "architect/num_params_total": num_params,
            "time/elapsed_s": round(time.time() - self._start_time, 1),
            **all_dag_metrics,
        }
        if extra_metrics:
            log_dict.update(extra_metrics)

        # Console summary
        print(f"  [Log] step={meta_step} task={task_name} rew={mean_rew:.3f} "
              f"loss={mean_loss:.4f} params={num_params:.0f} "
              f"action={action_taken} phase={phase}")

        if self.use_wandb:
            wandb.log(log_dict, step=meta_step)

    def log_dag_artifact(self, graphs, meta_step):
        """Log serialized DAG topologies as a WandB artifact."""
        dag_data = {name: serialize_dag_to_json(g) for name, g in graphs.items()}

        if self.use_wandb:
            artifact = wandb.Artifact(
                f"dag_topology_step_{meta_step}",
                type="dag_topology",
            )
            with artifact.new_file("topology.json") as f:
                f.write(json.dumps(dag_data, indent=2))
            wandb.log_artifact(artifact)

    def finish(self):
        elapsed = time.time() - self._start_time
        print(f"[Logger] Training finished in {elapsed:.1f}s")
        if self.use_wandb:
            wandb.finish()
