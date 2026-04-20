"""
Simulated Student using Bayesian Knowledge Tracing.

Three profiles:
  - beginner      : low prior, average learn rate, higher slip
  - intermediate  : medium prior, fast learn rate, low slip
  - gap_filled    : high prior on early concepts, low on advanced ones
                    (student skipped ahead somewhere)

Ground-truth mastery drives response probability; the TUTOR only sees its own
BKT-estimated mastery (what a real teacher would infer from observations).
"""

import numpy as np
from .bkt import BKTParams, bkt_update, expected_correct
from .question_bank import N_CONCEPTS, CONCEPTS, DIFFICULTY_PENALTY


PROFILES = {
    "beginner": {
        "p_init":  0.10,
        "p_learn": 0.12,
        "p_guess": 0.18,
        "p_slip":  0.12,
    },
    "intermediate": {
        "p_init":  0.35,
        "p_learn": 0.20,
        "p_guess": 0.20,
        "p_slip":  0.08,
    },
    "gap_filled": {
        # high prior on basics (0-3), low prior on advanced (4-9)
        "p_init_basic":    0.70,
        "p_init_advanced": 0.05,
        "p_learn": 0.15,
        "p_guess": 0.20,
        "p_slip":  0.10,
    },
}


def make_student(profile: str, seed: int = 0):
    rng = np.random.default_rng(seed)
    cfg = PROFILES[profile]

    params_per_concept = []
    true_mastery = np.zeros(N_CONCEPTS, dtype=np.float32)

    for c in range(N_CONCEPTS):
        if profile == "gap_filled":
            p_init = cfg["p_init_basic"] if c < 4 else cfg["p_init_advanced"]
            params_per_concept.append(BKTParams(
                p_init=p_init, p_learn=cfg["p_learn"],
                p_guess=cfg["p_guess"], p_slip=cfg["p_slip"]))
            # Ground truth: roughly matches prior but with some noise
            true_mastery[c] = float(rng.beta(p_init*10 + 1, (1-p_init)*10 + 1))
        else:
            params_per_concept.append(BKTParams(
                p_init=cfg["p_init"], p_learn=cfg["p_learn"],
                p_guess=cfg["p_guess"], p_slip=cfg["p_slip"]))
            true_mastery[c] = float(rng.beta(cfg["p_init"]*10 + 1,
                                             (1-cfg["p_init"])*10 + 1))

    return Student(profile, params_per_concept, true_mastery, rng)


class Student:
    def __init__(self, profile, params_per_concept, true_mastery, rng):
        self.profile = profile
        self.params = params_per_concept
        self.true_mastery = true_mastery.copy()
        self.rng = rng

    def prereq_bonus(self, concept: int) -> float:
        """Learning is faster when prereqs are mastered. Returns multiplier on p_learn."""
        prereqs = CONCEPTS[concept]["prereqs"]
        if not prereqs:
            return 1.0
        avg = np.mean([self.true_mastery[p] for p in prereqs])
        # If prereqs are weak, learning is slow (down to 0.3x)
        return 0.3 + 0.7 * avg

    def answer(self, concept: int, difficulty: str) -> bool:
        """Simulate an answer and update TRUE mastery (learning occurs)."""
        p = self.params[concept]
        penalty = DIFFICULTY_PENALTY[difficulty]
        p_correct = expected_correct(self.true_mastery[concept], p, penalty)
        correct = self.rng.random() < p_correct

        # True mastery update: attempting a problem teaches something regardless,
        # but prereq gaps slow learning.
        effective_learn = p.p_learn * self.prereq_bonus(concept)
        # Harder questions teach more when answered, but less when failed badly
        diff_boost = {"easy": 0.8, "medium": 1.0, "hard": 1.3}[difficulty]
        gain = effective_learn * diff_boost * (1.0 if correct else 0.5)
        new_true = self.true_mastery[concept] + (1 - self.true_mastery[concept]) * gain
        self.true_mastery[concept] = float(np.clip(new_true, 0.0, 1.0))

        return bool(correct)

    def total_mastery(self) -> float:
        return float(self.true_mastery.sum())
