"""
TaskSampler — Rotates through a pool of MiniGrid environments for meta-RL.
Each task gets a unique ID and one-hot embedding for the configurator.
"""
import numpy as np
from envs.wrappers import make_env

# Pool of MiniGrid tasks with varying difficulty and structure
DEFAULT_TASK_POOL = [
    "MiniGrid-Empty-6x6-v0",
    "MiniGrid-DoorKey-5x5-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "MiniGrid-GoToObject-8x8-N2-v0",
    "MiniGrid-LavaCrossingS9N1-v0",
    "MiniGrid-SimpleCrossingS9N1-v0",
]


class TaskSampler:
    """
    Manages a pool of MiniGrid environments.
    Provides task IDs, one-hot embeddings, and environment instances.
    """
    def __init__(self, task_pool=None, seed=42):
        self.task_pool = task_pool or DEFAULT_TASK_POOL
        self.num_tasks = len(self.task_pool)
        self.rng = np.random.RandomState(seed)
        self._envs = {}  # Lazy-initialized cache

    def sample_task_id(self):
        """Sample a random task index."""
        return self.rng.randint(0, self.num_tasks)

    def get_env(self, task_id):
        """Get or create the environment for a given task ID."""
        if task_id not in self._envs:
            self._envs[task_id] = make_env(self.task_pool[task_id])
        return self._envs[task_id]

    def get_task_name(self, task_id):
        return self.task_pool[task_id]

    def get_task_embedding(self, task_id):
        """One-hot encoding of the task ID."""
        emb = np.zeros(self.num_tasks, dtype=np.float32)
        emb[task_id] = 1.0
        return emb

    def close_all(self):
        for env in self._envs.values():
            env.close()
        self._envs.clear()
