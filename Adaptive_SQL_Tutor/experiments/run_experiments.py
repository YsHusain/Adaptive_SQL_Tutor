"""
Run all experiments: 4 agents x 3 student profiles, N episodes each.

For each (agent, profile) combination:
  - Train for TRAIN_EPISODES (learning agents only; baselines just act)
  - Evaluate on EVAL_EPISODES with exploration turned off
  - Log per-episode: total reward, final true mastery, time-to-mastery,
    per-step correctness, and action trajectory

Outputs:
  results/data/episode_log.csv       - per-episode metrics
  results/data/eval_summary.csv      - eval means +- std
  results/plots/learning_curves.png
  results/plots/final_mastery_bar.png
  results/plots/policy_heatmap_dqn.png
  results/plots/policy_heatmap_linucb.png
  results/plots/mastery_trajectory.png
"""
import argparse
import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# add project root to path
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.environment import TutorEnv
from src.question_bank import (
    CONCEPTS, N_CONCEPTS, DIFFICULTIES, action_to_concept_diff, num_actions,
)
from src.agents.random_agent import RandomAgent
from src.agents.fixed_curriculum import FixedCurriculumAgent
from src.agents.linucb_bandit import LinUCBAgent

from src.agents.dqn_agent import DQNAgent, backend_info as dqn_backend_info
HAS_DQN = True
print(f"[info] DQN backend: {dqn_backend_info()}")
from src.agents.session_manager import SessionManager
from src.agents.coordinator_agent import CoordinatorAgent
from src.agents.question_selector import QuestionSelector
from src.agents.hint_provider import HintProvider


RESULTS_DIR = os.path.join(ROOT, "results")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_episode(env, agent, seed, train=True):
    s = env.reset(seed=seed)
    total_r = 0.0
    n_correct = 0
    traj = []
    while True:
        a = agent.act(s)
        sp, r, d, info = env.step(a)
        if train:
            agent.update(s, a, r, sp, d)
        total_r += r
        n_correct += int(info["correct"])
        traj.append({
            "action": a,
            "concept": info["concept"],
            "difficulty": info["difficulty"],
            "correct": info["correct"],
            "est_mastery_sum": info["est_mastery_sum"],
            "true_mastery_sum": info["true_mastery_sum"],
        })
        s = sp
        if d:
            break
    return {
        "reward": total_r,
        "accuracy": n_correct / env.episode_length,
        "final_true_mastery": env.student.total_mastery(),
        "final_est_mastery": float(env.est_mastery.sum()),
        "trajectory": traj,
    }


def time_to_mastery(traj, threshold=0.8, n_mastered_req=5):
    """First step at which >= n_mastered_req concepts have est mastery >= threshold.
       Returns episode_length + 1 if never reached."""
    # We only have sums per step in trajectory; reconstruct via est vectors would be nicer
    # but using sum / N_CONCEPTS average as proxy, match final to sum:
    for i, step in enumerate(traj):
        # if average mastery exceeds threshold * n_mastered_req / N_CONCEPTS, consider met
        if step["est_mastery_sum"] >= threshold * n_mastered_req:
            return i + 1
    return len(traj) + 1


def make_agent(name, env, seed):
    n = env.n_actions
    if name == "random":
        return RandomAgent(n_actions=n, seed=seed)
    if name == "fixed_curriculum":
        return FixedCurriculumAgent(n_actions=n, seed=seed)
    if name == "linucb":
        return LinUCBAgent(n_actions=n, state_dim=env.state_dim, seed=seed)
    if name == "dqn" and HAS_DQN:
        return DQNAgent(n_actions=n, state_dim=env.state_dim, seed=seed)
    if name == "multi_agent":
        return MultiAgentAdapter(env=env, seed=seed)
    raise ValueError(name)


class MultiAgentAdapter:
    """Wraps SessionManager so it quacks like a single-agent RL interface
       (act/update/train_mode) for the experiment runner. Internally it runs
       the Coordinator -> QuestionSelector -> Hint Provider pipeline."""
    name = "multi_agent"

    def __init__(self, env, seed=0):
        self.env = env
        self.coord = CoordinatorAgent(state_dim=env.state_dim, seed=seed)
        self.selector = QuestionSelector(state_dim=env.state_dim, seed=seed)
        self.hinter = HintProvider()
        self._last_strategy = None  # cached for update()
        self._last_state_cache = None

    def train_mode(self, on=True):
        self.coord.train_mode(on)
        self.selector.train_mode(on)

    def act(self, state):
        strategy = self.coord.act(state)
        self._last_strategy = strategy
        action = self.selector.act(state, strategy)
        return action

    def update(self, state, action, reward, next_state, done):
        # Both agents share the same reward signal
        self.coord.update(state, self._last_strategy, reward, next_state, done)
        self.selector.update(state, action, reward)
        # Update coord's recent-accuracy context using the env-reported outcome.
        # The env already stepped; we recover 'correct' from info via the shortcut
        # that reward > 0 on mastery gain only loosely tracks it — instead we
        # sample from the env's history which is authoritative.
        if self.env.history:
            self.coord.observe_outcome(self.env.history[-1]["correct"])


def collect_policy_grid(agent, env, grid_size=5):
    """For visualization: sweep a 'basics vs advanced' mastery grid, record
       the difficulty the agent picks for a representative advanced concept."""
    grid = np.zeros((grid_size, grid_size))
    # axis 0: basics mastery (concepts 0-3), axis 1: advanced mastery (concepts 4-9)
    for i in range(grid_size):
        for j in range(grid_size):
            basics = i / (grid_size - 1)
            advanced = j / (grid_size - 1)
            state = np.zeros(N_CONCEPTS, dtype=np.float32)
            state[:4] = basics
            state[4:] = advanced
            # Turn off exploration for policy probing
            if hasattr(agent, "train_mode"):
                agent.train_mode(False)
            a = agent.act(state)
            if hasattr(agent, "train_mode"):
                agent.train_mode(True)
            concept, diff = action_to_concept_diff(a)
            # Encode: concept id * 3 + difficulty index, but for visualization
            # we'll plot the chosen CONCEPT (which matters most pedagogically)
            grid[i, j] = concept
    return grid


def run_all(train_episodes, eval_episodes, seeds):
    agents_to_run = ["random", "fixed_curriculum", "linucb"]
    if HAS_DQN:
        agents_to_run.append("dqn")
    agents_to_run.append("multi_agent")

    profiles = ["beginner", "intermediate", "gap_filled"]

    all_rows = []
    policy_grids = {}

    for profile in profiles:
        for agent_name in agents_to_run:
            for seed in seeds:
                print(f"\n=== {agent_name} | {profile} | seed {seed} ===")
                env = TutorEnv(profile=profile, episode_length=30, seed=seed)
                agent = make_agent(agent_name, env, seed=seed)

                t0 = time.time()
                # --- Training phase ---
                for ep in range(train_episodes):
                    res = run_episode(env, agent, seed=seed * 1000 + ep, train=True)
                    if (ep + 1) % max(1, train_episodes // 5) == 0 or ep == 0:
                        print(f"  train ep {ep+1:>4}/{train_episodes} | "
                              f"reward={res['reward']:7.2f} | "
                              f"final_mastery={res['final_true_mastery']:5.2f}")
                    all_rows.append({
                        "phase": "train",
                        "agent": agent_name, "profile": profile, "seed": seed,
                        "episode": ep, "reward": res["reward"],
                        "accuracy": res["accuracy"],
                        "final_true_mastery": res["final_true_mastery"],
                        "final_est_mastery": res["final_est_mastery"],
                        "time_to_mastery": time_to_mastery(res["trajectory"]),
                    })

                # --- Evaluation phase (no exploration/learning) ---
                if hasattr(agent, "train_mode"):
                    agent.train_mode(False)
                for ep in range(eval_episodes):
                    res = run_episode(env, agent, seed=10_000 + seed * 1000 + ep,
                                      train=False)
                    all_rows.append({
                        "phase": "eval",
                        "agent": agent_name, "profile": profile, "seed": seed,
                        "episode": ep, "reward": res["reward"],
                        "accuracy": res["accuracy"],
                        "final_true_mastery": res["final_true_mastery"],
                        "final_est_mastery": res["final_est_mastery"],
                        "time_to_mastery": time_to_mastery(res["trajectory"]),
                    })

                # Save policy grid only for learning agents on first seed
                if agent_name in ("linucb", "dqn") and seed == seeds[0]:
                    policy_grids[(agent_name, profile)] = collect_policy_grid(agent, env)

                print(f"  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(DATA_DIR, "episode_log.csv"), index=False)

    # Eval summary
    eval_df = df[df["phase"] == "eval"]
    summary = eval_df.groupby(["profile", "agent"]).agg(
        reward_mean=("reward", "mean"),
        reward_std=("reward", "std"),
        final_true_mastery_mean=("final_true_mastery", "mean"),
        final_true_mastery_std=("final_true_mastery", "std"),
        accuracy_mean=("accuracy", "mean"),
        time_to_mastery_mean=("time_to_mastery", "mean"),
    ).reset_index()
    summary.to_csv(os.path.join(DATA_DIR, "eval_summary.csv"), index=False)
    print("\n=== EVAL SUMMARY ===")
    print(summary.to_string(index=False))

    return df, summary, policy_grids


def plot_learning_curves(df):
    """Reward over training episodes, averaged across seeds, one subplot per profile."""
    train_df = df[df["phase"] == "train"]
    profiles = train_df["profile"].unique()
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, profiles):
        sub = train_df[train_df["profile"] == profile]
        for agent_name in sorted(sub["agent"].unique()):
            g = sub[sub["agent"] == agent_name]
            # rolling mean per seed, then average
            window = 20
            pivot = g.pivot_table(index="episode", columns="seed", values="reward")
            rolled = pivot.rolling(window=window, min_periods=1).mean()
            mean = rolled.mean(axis=1)
            std = rolled.std(axis=1)
            ax.plot(mean.index, mean.values, label=agent_name)
            ax.fill_between(mean.index, (mean - std).values, (mean + std).values, alpha=0.15)
        ax.set_title(f"Profile: {profile}")
        ax.set_xlabel("Training episode")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(f"Episode reward (rolling mean)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Learning Curves: Reward vs Training Episode")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "learning_curves.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def plot_final_mastery(summary):
    """Bar chart: final true mastery per agent per profile, with error bars."""
    profiles = summary["profile"].unique()
    agents = sorted(summary["agent"].unique())
    x = np.arange(len(profiles))
    width = 0.8 / len(agents)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, agent in enumerate(agents):
        sub = summary[summary["agent"] == agent].set_index("profile").reindex(profiles)
        ax.bar(x + i * width, sub["final_true_mastery_mean"], width,
               yerr=sub["final_true_mastery_std"], label=agent, capsize=3)
    ax.set_xticks(x + width * (len(agents) - 1) / 2)
    ax.set_xticklabels(profiles)
    ax.set_ylabel("Final true mastery (sum over 10 concepts)")
    ax.set_title("Final Mastery by Agent and Profile (eval episodes)")
    ax.axhline(10, color="gray", linestyle="--", alpha=0.4, label="max possible")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "final_mastery_bar.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def plot_policy_heatmap(policy_grids):
    """For each learning agent, show which concept it picks across a mastery grid."""
    if not policy_grids:
        return
    for (agent_name, profile), grid in policy_grids.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(grid, origin="lower", cmap="viridis",
                       vmin=0, vmax=N_CONCEPTS - 1)
        ax.set_xlabel("Advanced-concept mastery  (concepts 4-9)")
        ax.set_ylabel("Basic-concept mastery  (concepts 0-3)")
        ax.set_title(f"Policy heatmap: {agent_name} on {profile}\n"
                     f"(cell = concept chosen given mastery level)")
        cbar = fig.colorbar(im, ax=ax, ticks=range(N_CONCEPTS))
        cbar.ax.set_yticklabels([c["name"] for c in CONCEPTS])
        fig.tight_layout()
        out = os.path.join(PLOTS_DIR, f"policy_heatmap_{agent_name}_{profile}.png")
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved {out}")


def plot_mastery_trajectory(df):
    """For eval phase, show how final mastery builds up (mean across seeds) per agent."""
    # Already have final true mastery, but we want step-wise. Re-run one eval ep per
    # (agent, profile) with trajectory logging.
    from src.agents.random_agent import RandomAgent  # noqa
    profiles = df["profile"].unique()
    agents = sorted(df["agent"].unique())
    fig, axes = plt.subplots(1, len(profiles), figsize=(5 * len(profiles), 4), sharey=True)
    if len(profiles) == 1:
        axes = [axes]
    for ax, profile in zip(axes, profiles):
        for agent_name in agents:
            env = TutorEnv(profile=profile, episode_length=30, seed=777)
            agent = make_agent(agent_name, env, seed=777)
            # quick training burst so learning agents have something
            if agent_name in ("linucb", "dqn"):
                for ep in range(60):
                    run_episode(env, agent, seed=ep, train=True)
                if hasattr(agent, "train_mode"):
                    agent.train_mode(False)
            # 3 eval episodes -> average trajectory
            trajs = []
            for ep in range(3):
                res = run_episode(env, agent, seed=9000 + ep, train=False)
                trajs.append([s["true_mastery_sum"] for s in res["trajectory"]])
            mean_traj = np.mean(trajs, axis=0)
            ax.plot(range(1, len(mean_traj) + 1), mean_traj, label=agent_name)
        ax.set_xlabel("Step within episode")
        ax.set_title(f"Profile: {profile}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("True mastery (sum)")
    axes[-1].legend(loc="lower right")
    fig.suptitle("Mastery Trajectory Within Episode")
    fig.tight_layout()
    out = os.path.join(PLOTS_DIR, "mastery_trajectory.png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"Saved {out}")


def statistical_tests(df):
    """Welch's t-tests on eval final mastery vs fixed_curriculum, with
       Cohen's d effect size and 95% CI on the mean difference."""
    from scipy import stats
    eval_df = df[df["phase"] == "eval"]
    rows = []
    for profile in eval_df["profile"].unique():
        base = eval_df[(eval_df["profile"] == profile) &
                       (eval_df["agent"] == "fixed_curriculum")]["final_true_mastery"].values
        for agent in eval_df["agent"].unique():
            if agent == "fixed_curriculum":
                continue
            other = eval_df[(eval_df["profile"] == profile) &
                            (eval_df["agent"] == agent)]["final_true_mastery"].values
            if len(other) < 2 or len(base) < 2:
                continue
            t, p = stats.ttest_ind(other, base, equal_var=False)
            # Cohen's d with pooled SD
            pooled_sd = np.sqrt((other.var(ddof=1) + base.var(ddof=1)) / 2)
            cohen_d = (other.mean() - base.mean()) / (pooled_sd + 1e-12)
            # 95% CI on the mean difference (Welch-Satterthwaite)
            se = np.sqrt(other.var(ddof=1) / len(other) + base.var(ddof=1) / len(base))
            df_welch = (se ** 4) / (
                (other.var(ddof=1) / len(other)) ** 2 / (len(other) - 1) +
                (base.var(ddof=1) / len(base)) ** 2 / (len(base) - 1) + 1e-12
            )
            t_crit = stats.t.ppf(0.975, df_welch)
            md = float(other.mean() - base.mean())
            ci_lo, ci_hi = md - t_crit * se, md + t_crit * se
            rows.append({
                "profile": profile, "agent": agent, "vs": "fixed_curriculum",
                "mean_diff": md,
                "ci95_lo": float(ci_lo), "ci95_hi": float(ci_hi),
                "cohen_d": float(cohen_d),
                "t_stat": float(t), "p_value": float(p),
                "n_agent": len(other), "n_base": len(base),
            })
    tdf = pd.DataFrame(rows)
    tdf.to_csv(os.path.join(DATA_DIR, "stat_tests.csv"), index=False)
    print("\n=== STAT TESTS (vs fixed_curriculum) ===")
    print(tdf.to_string(index=False))
    return tdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_episodes", type=int, default=150,
                    help="Training episodes per (agent, profile, seed)")
    ap.add_argument("--eval_episodes", type=int, default=30,
                    help="Evaluation episodes per (agent, profile, seed)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                    help="Random seeds to run")
    ap.add_argument("--quick", action="store_true",
                    help="Quick smoke test: 10 train episodes, 5 eval, 1 seed")
    args = ap.parse_args()

    if args.quick:
        args.train_episodes, args.eval_episodes, args.seeds = 10, 5, [0]

    print(f"Config: train={args.train_episodes}, eval={args.eval_episodes}, "
          f"seeds={args.seeds}, dqn_backend={dqn_backend_info()}")

    df, summary, policy_grids = run_all(
        args.train_episodes, args.eval_episodes, args.seeds
    )
    plot_learning_curves(df)
    plot_final_mastery(summary)
    plot_policy_heatmap(policy_grids)
    plot_mastery_trajectory(df)

    try:
        statistical_tests(df)
    except Exception as e:
        print(f"[warn] stat tests skipped: {e}")

    print("\nAll outputs in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
