"""Random baseline: uniform random over (concept, difficulty)."""
import numpy as np


class RandomAgent:
    name = "random"

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, state):
        return int(self.rng.integers(0, self.n_actions))

    def update(self, *args, **kwargs):
        pass

    def train_mode(self, on=True): pass
