"""
Bayesian Knowledge Tracing (BKT) update logic.

For each concept, we maintain P(mastery) and update it after each observation
(correct/incorrect answer) using the standard BKT update rule.

BKT parameters per concept:
  p_init  : prior probability of mastery before any interaction
  p_learn : probability of transitioning from unmastered -> mastered per opportunity
  p_guess : probability of answering correctly when unmastered
  p_slip  : probability of answering incorrectly when mastered
"""

import numpy as np


class BKTParams:
    def __init__(self, p_init=0.2, p_learn=0.15, p_guess=0.2, p_slip=0.1):
        self.p_init = p_init
        self.p_learn = p_learn
        self.p_guess = p_guess
        self.p_slip = p_slip


def bkt_update(mastery_prior: float, correct: bool, params: BKTParams) -> float:
    """
    Standard BKT posterior update.

    Step 1: P(mastery | observation) via Bayes rule
    Step 2: P(mastery after opportunity to learn)
    """
    pL = mastery_prior
    if correct:
        num = pL * (1 - params.p_slip)
        den = pL * (1 - params.p_slip) + (1 - pL) * params.p_guess
    else:
        num = pL * params.p_slip
        den = pL * params.p_slip + (1 - pL) * (1 - params.p_guess)

    posterior = num / (den + 1e-12)
    # Learning opportunity: can transition from unmastered to mastered
    new_mastery = posterior + (1 - posterior) * params.p_learn
    return float(np.clip(new_mastery, 0.0, 1.0))


def expected_correct(mastery: float, params: BKTParams, difficulty_penalty: float = 0.0) -> float:
    """
    Probability a student with given mastery answers a question correctly.
    difficulty_penalty in [0, 0.3] roughly: harder questions increase effective slip
    and reduce effective guess rate.
    """
    eff_slip = min(1.0, params.p_slip + difficulty_penalty)
    eff_guess = max(0.0, params.p_guess - difficulty_penalty * 0.5)
    return mastery * (1 - eff_slip) + (1 - mastery) * eff_guess
