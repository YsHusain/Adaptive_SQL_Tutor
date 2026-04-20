"""
Live CLI demo: watch a trained agent teach a simulated student, step by step.

Usage:
    python demo/live_demo.py --agent linucb --profile beginner
    python demo/live_demo.py --agent dqn --profile gap_filled --train_episodes 200
    python demo/live_demo.py --agent fixed_curriculum --profile intermediate

This is what you'd screen-record for the 10-minute video. Shows the agent
adapting difficulty and topic selection based on the student's BKT state.
"""
import argparse
import os
import sys
import time

# Path setup so the demo runs from anywhere
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from src.environment import TutorEnv
from src.question_bank import CONCEPTS, get_question, action_to_concept_diff
from src.agents.random_agent import RandomAgent
from src.agents.fixed_curriculum import FixedCurriculumAgent
from src.agents.linucb_bandit import LinUCBAgent

from src.agents.dqn_agent import DQNAgent, backend_info as dqn_backend_info


def mastery_bar(values, width=30):
    """Render a horizontal bar for a single mastery value in [0,1]."""
    out = []
    for v in values:
        n = int(round(v * width))
        bar = "█" * n + "·" * (width - n)
        out.append(f"{bar} {v:.2f}")
    return out


def render_student_state(est_mastery, true_mastery):
    print("\n  Concept             | Tutor estimate                    | True mastery")
    print("  " + "-" * 88)
    for i, c in enumerate(CONCEPTS):
        est_bar = ("█" * int(round(est_mastery[i] * 20))).ljust(20, "·")
        true_bar = ("█" * int(round(true_mastery[i] * 20))).ljust(20, "·")
        print(f"  {c['name']:<18}  | {est_bar} {est_mastery[i]:.2f}  "
              f"| {true_bar} {true_mastery[i]:.2f}")


def build_and_train(agent_name, profile, train_episodes, seed):
    env = TutorEnv(profile=profile, episode_length=30, seed=seed)
    n = env.n_actions
    if agent_name == "random":
        agent = RandomAgent(n_actions=n, seed=seed)
    elif agent_name == "fixed_curriculum":
        agent = FixedCurriculumAgent(n_actions=n, seed=seed)
    elif agent_name == "linucb":
        agent = LinUCBAgent(n_actions=n, state_dim=env.state_dim, seed=seed)
    elif agent_name == "dqn":
        agent = DQNAgent(n_actions=n, state_dim=env.state_dim, seed=seed)
    else:
        raise ValueError(agent_name)

    if agent_name in ("linucb", "dqn") and train_episodes > 0:
        print(f"Training {agent_name} for {train_episodes} episodes "
              f"on '{profile}' profile...")
        for ep in range(train_episodes):
            s = env.reset(seed=seed * 1000 + ep)
            while True:
                a = agent.act(s)
                sp, r, d, info = env.step(a)
                agent.update(s, a, r, sp, d)
                s = sp
                if d:
                    break
            if (ep + 1) % max(1, train_episodes // 5) == 0:
                print(f"  trained {ep + 1}/{train_episodes}")
        print("Training complete.\n")

    if hasattr(agent, "train_mode"):
        agent.train_mode(False)
    return env, agent


def run_demo(agent_name, profile, train_episodes, demo_steps, pause, seed):
    env, agent = build_and_train(agent_name, profile, train_episodes, seed)

    # Reset for the demo session with a FRESH student
    s = env.reset(seed=seed + 9999)

    print("=" * 90)
    print(f"  ADAPTIVE SQL TUTOR - Live Demo")
    print(f"  Agent: {agent_name}   |   Student profile: {profile}   |   "
          f"Episode length: {demo_steps}")
    print("=" * 90)
    print("\nInitial state:")
    render_student_state(env.est_mastery, env.student.true_mastery)
    if pause > 0:
        time.sleep(pause)

    total_correct = 0
    for step in range(demo_steps):
        a = agent.act(s)
        concept, difficulty = action_to_concept_diff(a)
        question_text = get_question(concept, difficulty, idx=step)

        print("\n" + "=" * 90)
        print(f"  Step {step + 1:>2}/{demo_steps}")
        print(f"  Tutor picks -> Concept: {CONCEPTS[concept]['name']:<18}  "
              f"Difficulty: {difficulty}")
        print(f"  Question   : {question_text}")

        sp, r, d, info = env.step(a)
        outcome = "CORRECT" if info["correct"] else "WRONG  "
        total_correct += int(info["correct"])
        print(f"  Student    : {outcome}   "
              f"(reward={r:+.2f}, running accuracy={total_correct/(step+1):.1%})")

        render_student_state(env.est_mastery, env.student.true_mastery)

        s = sp
        if pause > 0:
            time.sleep(pause)
        if d:
            break

    print("\n" + "=" * 90)
    print("  SESSION SUMMARY")
    print("=" * 90)
    print(f"  Agent                 : {agent_name}")
    print(f"  Student profile       : {profile}")
    print(f"  Steps                 : {demo_steps}")
    print(f"  Accuracy              : {total_correct / demo_steps:.1%}  "
          f"(ZPD target ~75%)")
    print(f"  Final true mastery    : {env.student.total_mastery():.2f} / 10.00")
    print(f"  Final est mastery     : {env.est_mastery.sum():.2f} / 10.00")
    print(f"  Concepts >= 0.8 true  : "
          f"{int((env.student.true_mastery >= 0.8).sum())}/10")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", choices=["random", "fixed_curriculum", "linucb", "dqn"],
                    default="linucb")
    ap.add_argument("--profile", choices=["beginner", "intermediate", "gap_filled"],
                    default="beginner")
    ap.add_argument("--train_episodes", type=int, default=150,
                    help="Training episodes for learning agents before demo")
    ap.add_argument("--demo_steps", type=int, default=20)
    ap.add_argument("--pause", type=float, default=0.0,
                    help="Seconds to pause between steps (use for video recording)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    run_demo(args.agent, args.profile, args.train_episodes,
             args.demo_steps, args.pause, args.seed)


if __name__ == "__main__":
    main()
