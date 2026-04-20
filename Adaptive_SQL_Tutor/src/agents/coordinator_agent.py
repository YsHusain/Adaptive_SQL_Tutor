"""
Coordinator Agent — high-level pedagogical decision maker.

Decides the *strategy* for the next interaction (what kind of teaching action
to take), before the Question Selector picks the specific concept/difficulty.

Action space (4 strategies):
    0 = TEACH    — introduce a new, low-mastery concept (mastery < 0.3)
    1 = PRACTICE — reinforce a partially-mastered concept (0.3 ≤ mastery < 0.8)
    2 = REVIEW   — revisit a mastered concept at higher difficulty (mastery ≥ 0.8)
    3 = ASSESS   — probe an unknown/uncertain concept (high variance in estimate)

Learning algorithm: numpy DQN (reuses the same infrastructure as the single-agent
DQN but with a smaller 4-action output). The state is the 10-d BKT mastery vector
plus a 3-d context summary: [avg_mastery, n_mastered_concepts, recent_accuracy].
"""
import random
from collections import deque

import numpy as np


class _MLP:
    """2-layer MLP with ReLU, manual forward/backward (no torch dependency)."""
    def __init__(self, in_dim, out_dim, hidden=64, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0 / in_dim),
                             (in_dim, hidden)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden),
                             (hidden, hidden)).astype(np.float32)
        self.b2 = np.zeros(hidden, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden),
                             (hidden, out_dim)).astype(np.float32)
        self.b3 = np.zeros(out_dim, dtype=np.float32)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(z2, 0.0)
        q = h2 @ self.W3 + self.b3
        return q, (x, z1, h1, z2, h2)

    def backward(self, dq, cache):
        x, z1, h1, z2, h2 = cache
        B = x.shape[0]
        dW3 = (h2.T @ dq) / B
        db3 = dq.mean(axis=0)
        dh2 = dq @ self.W3.T
        dz2 = dh2 * (z2 > 0).astype(np.float32)
        dW2 = (h1.T @ dz2) / B
        db2 = dz2.mean(axis=0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (z1 > 0).astype(np.float32)
        dW1 = (x.T @ dz1) / B
        db1 = dz1.mean(axis=0)
        return [dW1, db1, dW2, db2, dW3, db3]

    def copy_from(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()
        self.W3, self.b3 = other.W3.copy(), other.b3.copy()


class _Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.b1, self.b2, self.eps = b1, b2, eps
        self.t = 0
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mh = self.m[i] / (1 - self.b1 ** self.t)
            vh = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + self.eps)


# Pedagogical action constants (exported)
TEACH, PRACTICE, REVIEW, ASSESS = 0, 1, 2, 3
ACTION_NAMES = {TEACH: "TEACH", PRACTICE: "PRACTICE",
                REVIEW: "REVIEW", ASSESS: "ASSESS"}
N_STRATEGIES = 4


class CoordinatorAgent:
    """High-level RL agent deciding pedagogical strategy per step."""
    name = "coordinator"

    def __init__(self, state_dim, hidden=64, lr=1e-3, gamma=0.95,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=4000,
                 buffer_size=10000, batch_size=64,
                 target_update_every=200, seed=0):
        self.state_dim = state_dim + 3  # +3 context features
        self.n_actions = N_STRATEGIES
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps
        self.target_update_every = target_update_every

        random.seed(seed)
        self.rng = np.random.default_rng(seed)

        self.q = _MLP(self.state_dim, self.n_actions, hidden=hidden, seed=seed)
        self.q_tgt = _MLP(self.state_dim, self.n_actions, hidden=hidden, seed=seed + 1)
        self.q_tgt.copy_from(self.q)
        self.opt = _Adam(self.q.params(), lr=lr)

        self.buffer = deque(maxlen=buffer_size)
        self.total_steps = 0
        self._training = True
        self._recent_correct = deque(maxlen=5)

    def train_mode(self, on=True):
        self._training = on

    def _epsilon(self):
        if not self._training:
            return 0.0
        frac = min(1.0, self.total_steps / self.eps_decay_steps)
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _augment(self, mastery_vec):
        """Append 3 context features to the BKT mastery vector."""
        m = np.asarray(mastery_vec, dtype=np.float32)
        avg_m = float(m.mean())
        n_mastered = float((m >= 0.8).sum()) / len(m)
        recent_acc = (float(sum(self._recent_correct)) / len(self._recent_correct)
                      if self._recent_correct else 0.5)
        return np.concatenate([m, [avg_m, n_mastered, recent_acc]]).astype(np.float32)

    def observe_outcome(self, correct: bool):
        """Called by the session manager after each interaction so the
           coordinator has recent-accuracy context at decision time."""
        self._recent_correct.append(1 if correct else 0)

    def act(self, mastery_vec, return_reason=False):
        """Pick a pedagogical strategy. Optionally returns a short rationale."""
        state = self._augment(mastery_vec)
        if self.rng.random() < self._epsilon():
            action = int(self.rng.integers(0, self.n_actions))
            reason = "exploration (random strategy)"
        else:
            q, _ = self.q.forward(state.reshape(1, -1))
            action = int(np.argmax(q[0]))
            reason = f"greedy (Q = {q[0, action]:+.2f})"
        if return_reason:
            return action, reason, state
        return action

    def update(self, mastery_vec, action, reward, next_mastery_vec, done):
        if not self._training:
            return
        s = self._augment(mastery_vec)
        sp = self._augment(next_mastery_vec)
        self.buffer.append((s, int(action), float(reward), sp, bool(done)))
        self.total_steps += 1

        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        s_b, a_b, r_b, sp_b, d_b = zip(*batch)
        s_b = np.stack(s_b).astype(np.float32)
        a_b = np.asarray(a_b, dtype=np.int64)
        r_b = np.asarray(r_b, dtype=np.float32)
        sp_b = np.stack(sp_b).astype(np.float32)
        d_b = np.asarray(d_b, dtype=np.float32)

        q_next, _ = self.q_tgt.forward(sp_b)
        target = r_b + self.gamma * (1.0 - d_b) * q_next.max(axis=1)

        q_pred, cache = self.q.forward(s_b)
        q_taken = q_pred[np.arange(len(a_b)), a_b]

        err = q_taken - target
        huber = np.where(np.abs(err) <= 1.0, err, np.sign(err)).astype(np.float32)
        dq = np.zeros_like(q_pred)
        dq[np.arange(len(a_b)), a_b] = huber

        grads = self.q.backward(dq, cache)
        tot_norm = float(np.sqrt(sum((g * g).sum() for g in grads)))
        if tot_norm > 10.0:
            scale = 10.0 / (tot_norm + 1e-12)
            grads = [g * scale for g in grads]
        self.opt.step(grads)

        if self.total_steps % self.target_update_every == 0:
            self.q_tgt.copy_from(self.q)
