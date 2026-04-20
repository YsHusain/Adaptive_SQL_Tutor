"""
Session Manager — orchestrates the multi-agent system.

Per step:
    1. Coordinator picks a pedagogical strategy from the BKT state
    2. Question Selector picks a specific (concept, difficulty) given that strategy
    3. Environment executes the action; student answers
    4. BKT (inside environment) updates the mastery estimate
    5. If the student answered wrong, Hint Provider generates a contextual hint
    6. Both learning agents (coordinator + selector) update from reward

This is the "Agent Orchestration Systems" path from the assignment rubric.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .coordinator_agent import CoordinatorAgent, ACTION_NAMES
from .question_selector import QuestionSelector
from .hint_provider import HintProvider
from ..question_bank import action_to_concept_diff, CONCEPTS


@dataclass
class StepTrace:
    """Per-step record of what every agent did, for the Streamlit UI and logs."""
    step: int
    strategy: int
    strategy_name: str
    strategy_reason: str
    concept: int
    concept_name: str
    difficulty: str
    question: str
    correct: bool
    reward: float
    hint: Optional[str] = None
    hint_type: Optional[str] = None
    est_mastery: list = field(default_factory=list)
    true_mastery: list = field(default_factory=list)


class SessionManager:
    """Orchestrates all four agents for a single episode."""
    name = "multi_agent_session"

    def __init__(self, env, coordinator=None, selector=None, hint_provider=None, seed=0):
        self.env = env
        self.coord = coordinator or CoordinatorAgent(state_dim=env.state_dim, seed=seed)
        self.selector = selector or QuestionSelector(state_dim=env.state_dim, seed=seed)
        self.hinter = hint_provider or HintProvider()

    def train_mode(self, on=True):
        self.coord.train_mode(on)
        self.selector.train_mode(on)

    def run_episode(self, seed=None, train=True, trace=False):
        """Run one episode. Returns summary dict and optionally full StepTrace list."""
        state = self.env.reset(seed=seed)

        total_reward = 0.0
        n_correct = 0
        traces: List[StepTrace] = []

        for step in range(self.env.episode_length):
            # 1. Coordinator picks strategy
            strategy, strat_reason, _ = self.coord.act(state, return_reason=True)

            # 2. Question Selector picks (concept, difficulty)
            action, sel_reason = self.selector.act(state, strategy, return_reason=True)

            # 3. Environment executes
            from ..question_bank import get_question
            concept, difficulty = action_to_concept_diff(action)
            question_text = get_question(concept, difficulty, idx=step)

            next_state, reward, done, info = self.env.step(action)

            # 4. Observe outcome in coordinator so its context stays current
            self.coord.observe_outcome(info["correct"])

            # 5. Hint on wrong answer
            hint_text, hint_type = None, None
            if not info["correct"]:
                hint_text, hint_type = self.hinter.get_hint(
                    concept, state[concept], return_type=True)

            # 6. Both learning agents update
            if train:
                self.coord.update(state, strategy, reward, next_state, done)
                self.selector.update(state, action, reward)

            total_reward += reward
            n_correct += int(info["correct"])

            if trace:
                traces.append(StepTrace(
                    step=step + 1,
                    strategy=strategy,
                    strategy_name=ACTION_NAMES[strategy],
                    strategy_reason=f"{strat_reason} | selector: {sel_reason}",
                    concept=concept,
                    concept_name=CONCEPTS[concept]["name"],
                    difficulty=difficulty,
                    question=question_text,
                    correct=bool(info["correct"]),
                    reward=float(reward),
                    hint=hint_text,
                    hint_type=hint_type,
                    est_mastery=next_state.tolist(),
                    true_mastery=self.env.student.true_mastery.tolist(),
                ))

            state = next_state
            if done:
                break

        summary = {
            "reward": total_reward,
            "accuracy": n_correct / self.env.episode_length,
            "final_true_mastery": float(self.env.student.total_mastery()),
            "final_est_mastery": float(self.env.est_mastery.sum()),
        }
        if trace:
            return summary, traces
        return summary
