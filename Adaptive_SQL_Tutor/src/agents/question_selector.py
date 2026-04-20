"""
Question Selector Agent.

Given a high-level pedagogical strategy from the Coordinator
(TEACH / PRACTICE / REVIEW / ASSESS), picks a specific (concept, difficulty)
question to deliver.

Implementation: constrains the action space based on the strategy, then uses
LinUCB contextual bandit for selection within that constrained space. This
gives us two-level decision making with a clean division of labor.

Strategy -> admissible (concept, difficulty) pairs:
    TEACH:    concepts with est_mastery < 0.30, difficulty = easy
    PRACTICE: 0.30 <= est_mastery < 0.80, difficulty = easy or medium
              (picks harder as mastery grows)
    REVIEW:   est_mastery >= 0.80, difficulty = hard
    ASSESS:   concept with highest prediction uncertainty (|est - 0.5|),
              any difficulty
"""
import numpy as np

from ..question_bank import N_CONCEPTS, DIFFICULTIES, concept_diff_to_action
from .coordinator_agent import TEACH, PRACTICE, REVIEW, ASSESS


class QuestionSelector:
    """Strategy-constrained LinUCB bandit."""
    name = "question_selector"

    def __init__(self, state_dim, alpha=0.6, seed=0):
        self.state_dim = state_dim
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        # One (A, b) per flat action (30 total), same as single-agent LinUCB
        self.n_actions = N_CONCEPTS * len(DIFFICULTIES)
        self.A = [np.eye(self.state_dim) for _ in range(self.n_actions)]
        self.b = [np.zeros(self.state_dim) for _ in range(self.n_actions)]
        self._training = True

    def train_mode(self, on=True):
        self._training = on

    def _admissible_actions(self, strategy, mastery):
        """Return the subset of flat action indices allowed by the strategy."""
        allowed = []
        if strategy == TEACH:
            for c in range(N_CONCEPTS):
                if mastery[c] < 0.30:
                    allowed.append(concept_diff_to_action(c, "easy"))
            if not allowed:
                # fallback: easiest un-mastered concept
                c_min = int(np.argmin(mastery))
                allowed = [concept_diff_to_action(c_min, "easy")]
        elif strategy == PRACTICE:
            for c in range(N_CONCEPTS):
                if 0.30 <= mastery[c] < 0.80:
                    # choose difficulty based on current mastery within the band
                    diff = "easy" if mastery[c] < 0.50 else "medium"
                    allowed.append(concept_diff_to_action(c, diff))
                    # also allow medium when low in band for exploration
                    if mastery[c] < 0.50:
                        allowed.append(concept_diff_to_action(c, "medium"))
            if not allowed:
                # fallback: best-guess concept at medium difficulty
                c_best = int(np.argmax(mastery * (mastery < 0.80)))
                allowed = [concept_diff_to_action(c_best, "medium")]
        elif strategy == REVIEW:
            for c in range(N_CONCEPTS):
                if mastery[c] >= 0.80:
                    allowed.append(concept_diff_to_action(c, "hard"))
            if not allowed:
                # fallback: highest-mastery concept at medium
                c_top = int(np.argmax(mastery))
                allowed = [concept_diff_to_action(c_top, "medium")]
        elif strategy == ASSESS:
            # Pick concept where we're most uncertain (mastery closest to 0.5)
            uncertainty = -np.abs(mastery - 0.5)
            c_probe = int(np.argmax(uncertainty))
            # Medium-difficulty probe question for best signal
            allowed = [concept_diff_to_action(c_probe, d) for d in DIFFICULTIES]
        else:
            allowed = list(range(self.n_actions))
        return allowed

    def act(self, mastery_vec, strategy, return_reason=False):
        """Select a question from the admissible set for the given strategy."""
        mastery = np.asarray(mastery_vec, dtype=np.float64)
        allowed = self._admissible_actions(strategy, mastery)

        # LinUCB over the admissible subset
        ucbs = {}
        for a in allowed:
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            ucbs[a] = float(theta @ mastery + self.alpha * np.sqrt(mastery @ A_inv @ mastery))

        best = max(ucbs, key=ucbs.get)
        # Tie-break randomly
        max_val = ucbs[best]
        cands = [a for a, v in ucbs.items() if np.isclose(v, max_val)]
        chosen = int(self.rng.choice(cands)) if len(cands) > 1 else best

        if return_reason:
            from ..question_bank import action_to_concept_diff, CONCEPTS
            concept, diff = action_to_concept_diff(chosen)
            reason = (f"admissible={len(allowed)} options; chose "
                      f"{CONCEPTS[concept]['name']}@{diff} "
                      f"(UCB={ucbs[chosen]:+.2f})")
            return chosen, reason
        return chosen

    def update(self, mastery_vec, action, reward):
        if not self._training:
            return
        x = np.asarray(mastery_vec, dtype=np.float64)
        self.A[action] += np.outer(x, x)
        self.b[action] += reward * x
