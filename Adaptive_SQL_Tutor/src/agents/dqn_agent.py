"""
DQN agent.

Tries to use PyTorch first. If torch is unavailable or fails to load (e.g.
Windows DLL issues), falls back to a pure-numpy implementation with the same
interface. Slower than torch but trains in a reasonable time for 10-d state /
30-action problems like this one.

State:   BKT mastery vector (dim 10)
Action:  30 discrete (concept x difficulty) choices
Network: 2-layer MLP [10 -> 128 -> 128 -> 30]
Updates: DQN with experience replay + target network (Mnih et al. 2015)
         Huber loss, Adam optimizer, gradient clipping.
"""
import random
from collections import deque

import numpy as np


# --------------------------------------------------------------------------- #
# Try PyTorch backend first                                                   #
# --------------------------------------------------------------------------- #
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
    _TORCH_ERR = None
except Exception as e:
    _HAS_TORCH = False
    _TORCH_ERR = e


# --------------------------------------------------------------------------- #
# PyTorch implementation                                                      #
# --------------------------------------------------------------------------- #
if _HAS_TORCH:

    class _QNetTorch(nn.Module):
        def __init__(self, state_dim, n_actions, hidden=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_actions),
            )

        def forward(self, x):
            return self.net(x)


    class DQNAgent:
        name = "dqn"
        backend = "torch"

        def __init__(self, n_actions, state_dim, lr=1e-3, gamma=0.95,
                     eps_start=1.0, eps_end=0.05, eps_decay_steps=6000,
                     buffer_size=20000, batch_size=64,
                     target_update_every=200, seed=0):
            self.n_actions = n_actions
            self.state_dim = state_dim
            self.gamma = gamma
            self.batch_size = batch_size
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay_steps = eps_decay_steps
            self.target_update_every = target_update_every

            torch.manual_seed(seed)
            random.seed(seed)
            self.rng = np.random.default_rng(seed)

            self.q = _QNetTorch(state_dim, n_actions)
            self.q_target = _QNetTorch(state_dim, n_actions)
            self.q_target.load_state_dict(self.q.state_dict())
            self.opt = optim.Adam(self.q.parameters(), lr=lr)
            self.loss_fn = nn.SmoothL1Loss()

            self.buffer = deque(maxlen=buffer_size)
            self.total_steps = 0
            self._training = True

        def train_mode(self, on=True):
            self._training = on
            self.q.train(on)

        def _epsilon(self):
            if not self._training:
                return 0.0
            frac = min(1.0, self.total_steps / self.eps_decay_steps)
            return self.eps_start + frac * (self.eps_end - self.eps_start)

        def act(self, state):
            if self.rng.random() < self._epsilon():
                return int(self.rng.integers(0, self.n_actions))
            with torch.no_grad():
                s = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                q_vals = self.q(s).squeeze(0).numpy()
            return int(np.argmax(q_vals))

        def update(self, state, action, reward, next_state, done):
            if not self._training:
                return
            self.buffer.append((np.asarray(state, dtype=np.float32),
                                int(action), float(reward),
                                np.asarray(next_state, dtype=np.float32),
                                bool(done)))
            self.total_steps += 1

            if len(self.buffer) < self.batch_size:
                return

            batch = random.sample(self.buffer, self.batch_size)
            s, a, r, sp, d = zip(*batch)
            s = torch.as_tensor(np.stack(s), dtype=torch.float32)
            a = torch.as_tensor(a, dtype=torch.long).unsqueeze(1)
            r = torch.as_tensor(r, dtype=torch.float32).unsqueeze(1)
            sp = torch.as_tensor(np.stack(sp), dtype=torch.float32)
            d = torch.as_tensor(d, dtype=torch.float32).unsqueeze(1)

            q_pred = self.q(s).gather(1, a)
            with torch.no_grad():
                q_next = self.q_target(sp).max(1, keepdim=True).values
                target = r + self.gamma * (1 - d) * q_next

            loss = self.loss_fn(q_pred, target)
            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
            self.opt.step()

            if self.total_steps % self.target_update_every == 0:
                self.q_target.load_state_dict(self.q.state_dict())


# --------------------------------------------------------------------------- #
# Pure-numpy fallback implementation                                          #
# --------------------------------------------------------------------------- #
else:

    class _MLPNumpy:
        """2-layer MLP with ReLU activations, all manual forward/backward."""
        def __init__(self, state_dim, n_actions, hidden=128, seed=0):
            rng = np.random.default_rng(seed)
            # He initialization (sensible for ReLU networks)
            self.W1 = rng.normal(0, np.sqrt(2.0 / state_dim),
                                 (state_dim, hidden)).astype(np.float32)
            self.b1 = np.zeros(hidden, dtype=np.float32)
            self.W2 = rng.normal(0, np.sqrt(2.0 / hidden),
                                 (hidden, hidden)).astype(np.float32)
            self.b2 = np.zeros(hidden, dtype=np.float32)
            self.W3 = rng.normal(0, np.sqrt(2.0 / hidden),
                                 (hidden, n_actions)).astype(np.float32)
            self.b3 = np.zeros(n_actions, dtype=np.float32)

        def params(self):
            return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

        def forward(self, x):
            z1 = x @ self.W1 + self.b1
            h1 = np.maximum(z1, 0.0)
            z2 = h1 @ self.W2 + self.b2
            h2 = np.maximum(z2, 0.0)
            q = h2 @ self.W3 + self.b3
            cache = (x, z1, h1, z2, h2)
            return q, cache

        def backward(self, dq, cache):
            """dq shape (B, n_actions). Returns grads in params() order."""
            x, z1, h1, z2, h2 = cache
            B = x.shape[0]
            # Layer 3
            dW3 = (h2.T @ dq) / B
            db3 = dq.mean(axis=0)
            dh2 = dq @ self.W3.T
            # Layer 2
            dz2 = dh2 * (z2 > 0).astype(np.float32)
            dW2 = (h1.T @ dz2) / B
            db2 = dz2.mean(axis=0)
            dh1 = dz2 @ self.W2.T
            # Layer 1
            dz1 = dh1 * (z1 > 0).astype(np.float32)
            dW1 = (x.T @ dz1) / B
            db1 = dz1.mean(axis=0)
            return [dW1, db1, dW2, db2, dW3, db3]

        def copy_from(self, other):
            self.W1 = other.W1.copy(); self.b1 = other.b1.copy()
            self.W2 = other.W2.copy(); self.b2 = other.b2.copy()
            self.W3 = other.W3.copy(); self.b3 = other.b3.copy()


    class _AdamOpt:
        def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
            self.params = params
            self.lr = lr
            self.beta1 = beta1
            self.beta2 = beta2
            self.eps = eps
            self.t = 0
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        def step(self, grads):
            self.t += 1
            for i, (p, g) in enumerate(zip(self.params, grads)):
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g * g)
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                # in-place so self.params aliases remain valid
                p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


    class DQNAgent:
        name = "dqn"
        backend = "numpy"

        def __init__(self, n_actions, state_dim, lr=1e-3, gamma=0.95,
                     eps_start=1.0, eps_end=0.05, eps_decay_steps=6000,
                     buffer_size=20000, batch_size=64,
                     target_update_every=200, seed=0):
            self.n_actions = n_actions
            self.state_dim = state_dim
            self.gamma = gamma
            self.batch_size = batch_size
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay_steps = eps_decay_steps
            self.target_update_every = target_update_every

            random.seed(seed)
            self.rng = np.random.default_rng(seed)

            self.q = _MLPNumpy(state_dim, n_actions, hidden=128, seed=seed)
            self.q_target = _MLPNumpy(state_dim, n_actions, hidden=128,
                                      seed=seed + 1)
            self.q_target.copy_from(self.q)
            self.opt = _AdamOpt(self.q.params(), lr=lr)

            self.buffer = deque(maxlen=buffer_size)
            self.total_steps = 0
            self._training = True

        def train_mode(self, on=True):
            self._training = on

        def _epsilon(self):
            if not self._training:
                return 0.0
            frac = min(1.0, self.total_steps / self.eps_decay_steps)
            return self.eps_start + frac * (self.eps_end - self.eps_start)

        def act(self, state):
            if self.rng.random() < self._epsilon():
                return int(self.rng.integers(0, self.n_actions))
            x = np.asarray(state, dtype=np.float32).reshape(1, -1)
            q, _ = self.q.forward(x)
            return int(np.argmax(q[0]))

        def update(self, state, action, reward, next_state, done):
            if not self._training:
                return
            self.buffer.append((np.asarray(state, dtype=np.float32),
                                int(action), float(reward),
                                np.asarray(next_state, dtype=np.float32),
                                bool(done)))
            self.total_steps += 1

            if len(self.buffer) < self.batch_size:
                return

            batch = random.sample(self.buffer, self.batch_size)
            s, a, r, sp, d = zip(*batch)
            s_b = np.stack(s).astype(np.float32)
            a_b = np.asarray(a, dtype=np.int64)
            r_b = np.asarray(r, dtype=np.float32)
            sp_b = np.stack(sp).astype(np.float32)
            d_b = np.asarray(d, dtype=np.float32)

            # Target: r + gamma * max_a' Q_target(s', a') * (1-done)
            q_next, _ = self.q_target.forward(sp_b)
            q_next_max = q_next.max(axis=1)
            target = r_b + self.gamma * (1.0 - d_b) * q_next_max  # (B,)

            # Forward on online net
            q_pred, cache = self.q.forward(s_b)
            q_taken = q_pred[np.arange(len(a_b)), a_b]  # (B,)

            # Huber (Smooth L1) gradient
            error = q_taken - target  # (B,)
            huber_grad = np.where(np.abs(error) <= 1.0,
                                  error,
                                  np.sign(error)).astype(np.float32)

            # Distribute gradient only to the taken action
            dq = np.zeros_like(q_pred)
            dq[np.arange(len(a_b)), a_b] = huber_grad

            grads = self.q.backward(dq, cache)

            # Gradient clipping (global norm)
            total_norm = float(np.sqrt(sum((g * g).sum() for g in grads)))
            clip_at = 10.0
            if total_norm > clip_at:
                scale = clip_at / (total_norm + 1e-12)
                grads = [g * scale for g in grads]

            self.opt.step(grads)

            if self.total_steps % self.target_update_every == 0:
                self.q_target.copy_from(self.q)


def backend_info():
    """Return a human-readable string about which DQN backend is active."""
    if _HAS_TORCH:
        return "torch"
    return f"numpy (torch unavailable: {type(_TORCH_ERR).__name__})"
