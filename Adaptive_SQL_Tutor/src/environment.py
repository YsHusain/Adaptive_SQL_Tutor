"""
TutorEnv: Gym-style environment wrapping the student simulator.

State (what the AGENT observes): 10-d BKT-estimated mastery vector.
Action: flat id in [0, 30) -> (concept, difficulty) via question_bank.

Reward function is LEARNING-GAIN based (not accuracy), grounded in ZPD:
    r = mastery_gain                 # sum of BKT mastery improvement
      + zpd_bonus                    # small reward for staying in 0.7-0.85 success band
      - frustration_penalty          # 3+ consecutive failures
      - boredom_penalty              # targeting concepts with mastery > 0.95

This is the key differentiator called out in the plan.
"""

import numpy as np
from .bkt import BKTParams, bkt_update, expected_correct
from .student_simulator import make_student
from .question_bank import (
    N_CONCEPTS, CONCEPTS, DIFFICULTIES, DIFFICULTY_PENALTY,
    action_to_concept_diff, num_actions,
)


class TutorEnv:
    def __init__(self, profile: str = "beginner", episode_length: int = 30, seed: int = 0):
        self.profile = profile
        self.episode_length = episode_length
        self.seed = seed
        self._rng = np.random.default_rng(seed)
        # The TUTOR's BKT parameters (may differ from student's true params -
        # here we use a reasonable shared estimate; this is realistic)
        self.tutor_params = [BKTParams(p_init=0.25, p_learn=0.15,
                                       p_guess=0.20, p_slip=0.10)
                             for _ in range(N_CONCEPTS)]
        self.reset()

    @property
    def n_actions(self):
        return num_actions()

    @property
    def state_dim(self):
        return N_CONCEPTS

    def reset(self, seed: int = None):
        if seed is not None:
            self.seed = seed
            self._rng = np.random.default_rng(seed)
        self.student = make_student(self.profile, seed=int(self._rng.integers(0, 1_000_000)))
        # Tutor's estimate starts at prior
        self.est_mastery = np.array([p.p_init for p in self.tutor_params], dtype=np.float32)
        self.step_count = 0
        self.consec_fail = 0
        self.history = []  # list of dicts for analysis
        return self.est_mastery.copy()

    def step(self, action: int):
        concept, difficulty = action_to_concept_diff(action)

        mastery_before = self.est_mastery.sum()

        # Student answers (updates student's true mastery)
        correct = self.student.answer(concept, difficulty)

        # Tutor updates its BKT estimate for that concept
        self.est_mastery[concept] = bkt_update(
            self.est_mastery[concept], correct, self.tutor_params[concept]
        )

        mastery_after = self.est_mastery.sum()
        mastery_gain = mastery_after - mastery_before

        # ---- Reward engineering (ZPD-based) ----
        # Base: mastery gain (scaled)
        reward = 10.0 * mastery_gain

        # ZPD bonus: reward keeping student in productive struggle zone.
        # Estimate "expected success" using tutor's model:
        p_success_est = expected_correct(
            self.est_mastery[concept],
            self.tutor_params[concept],
            DIFFICULTY_PENALTY[difficulty],
        )
        if 0.70 <= p_success_est <= 0.85:
            reward += 0.3

        # Frustration: track consecutive failures
        if correct:
            self.consec_fail = 0
        else:
            self.consec_fail += 1
        if self.consec_fail >= 3:
            reward -= 0.5

        # Boredom: penalize choosing already-mastered concepts
        if mastery_before / N_CONCEPTS > 0.5 and self.est_mastery[concept] > 0.95 and difficulty != "hard":
            reward -= 0.4

        self.step_count += 1
        done = self.step_count >= self.episode_length

        self.history.append({
            "step": self.step_count,
            "concept": concept,
            "concept_name": CONCEPTS[concept]["name"],
            "difficulty": difficulty,
            "correct": correct,
            "est_mastery": self.est_mastery.copy(),
            "true_mastery": self.student.true_mastery.copy(),
            "reward": reward,
        })

        info = {
            "correct": correct,
            "concept": concept,
            "difficulty": difficulty,
            "true_mastery_sum": float(self.student.true_mastery.sum()),
            "est_mastery_sum": float(self.est_mastery.sum()),
            "p_success_est": p_success_est,
        }
        return self.est_mastery.copy(), float(reward), done, info
