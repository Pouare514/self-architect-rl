from envs.wrappers import make_env

def test():
    env = make_env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print("Test MiniGrid environment setup: SUCCESS")

if __name__ == "__main__":
    test()
