import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import numpy as np

class PyTorchObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, 
            shape=(old_shape[2], old_shape[0], old_shape[1]), 
            dtype=np.float32
        )

    def observation(self, observation):
        # HWC -> CHW, scale to [0, 1]
        obs = np.transpose(observation, (2, 0, 1)).astype(np.float32) / 255.0
        return obs

def make_env(env_id="MiniGrid-DoorKey-8x8-v0"):
    """
    Crée un environnement MiniGrid et applique les wrappers nécessaires
    (e.g., RGB observation, dict observation to flat/array, etc.)
    """
    env = gym.make(env_id, render_mode="rgb_array")
    # Convertit l'observation sous forme de grille symbolique en image RGB
    env = RGBImgObsWrapper(env)
    # Extrait uniquement l'image (ignore 'mission' et 'direction') pour simplifier le RL vision-based
    env = ImgObsWrapper(env)
    # Convertit en format PyTorch (C, H, W) float32
    env = PyTorchObsWrapper(env)
    
    return env
