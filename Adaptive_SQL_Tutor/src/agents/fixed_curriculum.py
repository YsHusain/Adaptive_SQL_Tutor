"""
Fixed curriculum baseline: teach concepts in dependency order at medium
difficulty. Advances to next concept once current estimated mastery > 0.8.
This is a strong rule-based baseline; it's what a human TA might code.
"""
import numpy as np
from ..question_bank import CONCEPTS, N_CONCEPTS, concept_diff_to_action


def topological_order():
    order, seen = [], set()
    def visit(c):
        if c in seen: return
        for p in CONCEPTS[c]["prereqs"]:
            visit(p)
        seen.add(c)
        order.append(c)
    for c in range(N_CONCEPTS):
        visit(c)
    return order


class FixedCurriculumAgent:
    name = "fixed_curriculum"

    def __init__(self, n_actions: int, seed: int = 0):
        self.n_actions = n_actions
        self.order = topological_order()

    def act(self, state):
        # state is the estimated mastery vector
        mastery = np.asarray(state)
        for c in self.order:
            if mastery[c] < 0.80:
                # pick difficulty based on current mastery
                if mastery[c] < 0.35:
                    diff = "easy"
                elif mastery[c] < 0.70:
                    diff = "medium"
                else:
                    diff = "hard"
                return concept_diff_to_action(c, diff)
        # everything mastered -> hard review on last concept
        return concept_diff_to_action(self.order[-1], "hard")

    def update(self, *args, **kwargs):
        pass

    def train_mode(self, on=True): pass
