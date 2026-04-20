"""
LinUCB contextual bandit.

Context: BKT-estimated mastery vector (state_dim = 10).
Arms:    30 flat actions (concept x difficulty).

For each arm a, we maintain:
    A_a = d x d matrix  (regularized covariance)
    b_a = d vector      (context * reward)
Then:
    theta_a = A_a^{-1} b_a
    ucb_a   = theta_a^T x + alpha * sqrt(x^T A_a^{-1} x)

This is a well-established method (Li et al. 2010) and converges fast.
"""
import numpy as np


class LinUCBAgent:
    name = "linucb"

    def __init__(self, n_actions: int, state_dim: int, alpha: float = 0.6, seed: int = 0):
        self.n_actions = n_actions
        self.d = state_dim
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.A = [np.eye(self.d) for _ in range(n_actions)]
        self.b = [np.zeros(self.d) for _ in range(n_actions)]
        self._training = True

    def train_mode(self, on=True):
        self._training = on

    def act(self, state):
        x = np.asarray(state, dtype=np.float64)
        ucbs = np.zeros(self.n_actions)
        for a in range(self.n_actions):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucb = float(theta @ x + self.alpha * np.sqrt(x @ A_inv @ x))
            ucbs[a] = ucb
        # tie-break randomly
        max_val = ucbs.max()
        candidates = np.where(np.isclose(ucbs, max_val))[0]
        return int(self.rng.choice(candidates))

    def update(self, state, action, reward, next_state=None, done=False):
        if not self._training:
            return
        x = np.asarray(state, dtype=np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
