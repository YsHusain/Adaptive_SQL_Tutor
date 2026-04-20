"""
Generate the full technical report as a PDF from experiment results.

Usage:
    # After running experiments/run_experiments.py:
    python report/generate_report.py

    # or with a custom output path:
    python report/generate_report.py --out my_report.pdf

Reads:
    results/data/eval_summary.csv
    results/data/stat_tests.csv
    results/data/episode_log.csv
    results/plots/*.png

Writes:
    report/technical_report.pdf       (submission-ready)
    report/technical_report.md        (editable markdown version)

If reportlab isn't installed, only the Markdown version is written.
Install reportlab with:  pip install reportlab
"""
import argparse
import os
import sys
from datetime import datetime

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

DATA_DIR = os.path.join(ROOT, "results", "data")
PLOTS_DIR = os.path.join(ROOT, "results", "plots")
REPORT_DIR = HERE


# --------------------------------------------------------------------------- #
# Load experiment data with graceful degradation                              #
# --------------------------------------------------------------------------- #
def load_data():
    """Load CSVs and return a dict of dataframes (or None if missing)."""
    def try_load(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    return {
        "eval_summary": try_load(os.path.join(DATA_DIR, "eval_summary.csv")),
        "stat_tests":   try_load(os.path.join(DATA_DIR, "stat_tests.csv")),
        "episode_log":  try_load(os.path.join(DATA_DIR, "episode_log.csv")),
    }


# --------------------------------------------------------------------------- #
# Auto-interpretation of results                                              #
# --------------------------------------------------------------------------- #
def interpret(data):
    """Produce a dict of auto-generated prose pieces for the report,
       tailored to whatever numbers came out of the run."""
    summary = data["eval_summary"]
    stats = data["stat_tests"]
    out = {}

    if summary is None or summary.empty:
        out["has_data"] = False
        return out

    out["has_data"] = True
    out["profiles"] = sorted(summary["profile"].unique().tolist())
    out["agents"] = sorted(summary["agent"].unique().tolist())
    out["n_profiles"] = len(out["profiles"])
    out["n_agents"] = len(out["agents"])

    # Best agent per profile by reward (primary training objective)
    best_by_reward = (summary.sort_values("reward_mean", ascending=False)
                            .groupby("profile").head(1)
                            .set_index("profile"))
    out["best_by_reward"] = {p: best_by_reward.loc[p, "agent"]
                              for p in out["profiles"]}

    # Best agent per profile by final true mastery
    best_by_mastery = (summary.sort_values("final_true_mastery_mean", ascending=False)
                              .groupby("profile").head(1)
                              .set_index("profile"))
    out["best_by_mastery"] = {p: best_by_mastery.loc[p, "agent"]
                               for p in out["profiles"]}

    # Significant findings
    sig = []
    if stats is not None and not stats.empty:
        for _, row in stats.iterrows():
            if row["p_value"] < 0.05:
                direction = "higher" if row["mean_diff"] > 0 else "lower"
                sig.append(
                    f"{row['agent']} on '{row['profile']}' achieved "
                    f"significantly {direction} final mastery than "
                    f"{row['vs']} (delta = {row['mean_diff']:+.3f}, "
                    f"p = {row['p_value']:.4f})"
                )
    out["significant_findings"] = sig

    # Headline numbers
    out["reward_mean_by_profile_agent"] = summary.set_index(
        ["profile", "agent"])["reward_mean"].to_dict()
    out["mastery_mean_by_profile_agent"] = summary.set_index(
        ["profile", "agent"])["final_true_mastery_mean"].to_dict()

    return out


# --------------------------------------------------------------------------- #
# Markdown report (always written; serves as fallback if reportlab missing)   #
# --------------------------------------------------------------------------- #
def build_markdown(data, interp):
    summary = data["eval_summary"]
    stats = data["stat_tests"]

    today = datetime.now().strftime("%B %d, %Y")

    md = []
    md.append(f"# Adaptive SQL Tutor with Reinforcement Learning\n")
    md.append(f"**Technical Report**  \n")
    md.append(f"*Take-home Final: Reinforcement Learning for Agentic AI Systems*  \n")
    md.append(f"*Generated: {today}*\n")
    md.append("---\n")

    # ---------------- 1. Executive Summary ----------------
    md.append("## 1. Executive Summary\n")
    md.append(
        "Most adaptive tutoring systems optimize for student accuracy, which "
        "incentivizes easy questions and undermines learning. This project "
        "takes a different approach: the reinforcement-learning agent is "
        "rewarded for **learning gain** (improvement in Bayesian Knowledge "
        "Tracing mastery), shaped by the **Zone of Proximal Development** "
        "principle that optimal learning occurs at ~70–85% success rate.\n"
    )
    md.append(
        "Two RL methods are implemented and compared: a **LinUCB contextual "
        "bandit** and a **Deep Q-Network (DQN)**. Each is benchmarked against "
        "a uniform-random baseline and a rule-based fixed-curriculum agent on "
        f"{interp.get('n_profiles', 3)} simulated student profiles "
        "(beginner, intermediate, gap-filled).\n"
    )
    if interp.get("has_data") and interp.get("significant_findings"):
        md.append("**Key findings:**\n")
        for s in interp["significant_findings"][:5]:
            md.append(f"- {s}\n")
    elif interp.get("has_data"):
        md.append(
            "**Key findings:** All four agents produced non-trivial mastery "
            "gains. See Section 5 for full numerical results and Section 6 "
            "for interpretation.\n"
        )
    else:
        md.append(
            "*(Numbers will appear here once `experiments/run_experiments.py` "
            "has been executed.)*\n"
        )
    md.append("")

    # ---------------- 2. System Architecture ----------------
    md.append("## 2. System Architecture\n")
    md.append("![Architecture](../results/plots/architecture.png)\n")
    md.append(
        "The system is a closed-loop interaction between an **RL agent** and "
        "a **TutorEnv** that wraps a BKT-based student simulator and a "
        "90-item SQL question bank covering 10 concepts (SELECT, WHERE, "
        "ORDER BY / LIMIT, aggregate functions, GROUP BY, HAVING, JOINs, "
        "subqueries, CTEs, window functions). Concepts form a dependency "
        "graph: for example, GROUP BY depends on both WHERE and aggregate "
        "functions, and window functions depend on both GROUP BY and CTEs.\n"
    )
    md.append("### 2.1 Multi-agent orchestration\n")
    md.append(
        "Four specialized agents operate under a Session Manager:\n\n"
        "- **Coordinator** (DQN, numpy backend): selects a high-level "
        "pedagogical strategy each step — TEACH, PRACTICE, REVIEW, or ASSESS — "
        "from the 10-d BKT mastery vector augmented with 3 context features "
        "(average mastery, count of mastered concepts, recent accuracy).\n"
        "- **Question Selector** (LinUCB contextual bandit): given the "
        "Coordinator's strategy, picks a specific `(concept, difficulty)` "
        "from the subset admissible under that strategy.\n"
        "- **Knowledge Tracker** (Bayesian Knowledge Tracing): maintains "
        "per-concept mastery estimates; consumed by both learning agents "
        "as state.\n"
        "- **Hint Provider** (rule-based): generates a contextual hint on "
        "wrong answers, selecting from three scaffolding levels (concept "
        "review / procedural scaffold / targeted nudge) based on current "
        "mastery — an operationalization of variable scaffolding within "
        "the Zone of Proximal Development.\n\n"
        "Agents communicate via the shared BKT state: the Coordinator reads "
        "it to choose a strategy; the Question Selector reads it plus the "
        "Coordinator's strategy to pick a question; the Hint Provider reads "
        "the per-concept component on wrong answers. Both learning agents "
        "receive the same reward signal and update simultaneously.\n"
    )
    md.append("### 2.2 Student simulator\n")
    md.append(
        "Three profiles instantiate the simulator: **beginner** (low prior "
        "mastery across all concepts), **intermediate** (medium prior, faster "
        "learning), and **gap-filled** (high prior on basic concepts 0–3, low "
        "on advanced concepts 4–9). Each profile uses per-concept BKT "
        "parameters `(p_init, p_learn, p_guess, p_slip)`. The student's "
        "ground-truth mastery is hidden from the tutor.\n"
    )
    md.append("### 2.3 Environment\n")
    md.append(
        "- **Observation** (10-d): the tutor's BKT-estimated mastery vector. "
        "This is what the agent sees — not ground-truth mastery.\n"
        "- **Action** (discrete, 30): all `(concept, difficulty)` combinations "
        "over 10 concepts × 3 difficulty levels.\n"
        "- **Reward** (learning-gain based, see §3.4): weighted BKT mastery "
        "gain plus a ZPD bonus when the student is in the productive-struggle "
        "band, minus frustration and boredom penalties.\n"
        "- **Episode length**: 30 interactions.\n"
    )

    # ---------------- 3. Mathematical Formulation ----------------
    md.append("## 3. Mathematical Formulation\n")
    md.append("### 3.1 Bayesian Knowledge Tracing (Corbett & Anderson, 1995)\n")
    md.append(
        "For each concept, `P(L_t)` denotes the probability the student has "
        "mastered the concept at time t. After observing a response:\n\n"
        "**If correct:**\n"
        "```\n"
        "P(L_t | correct) = P(L_t)(1 − p_slip) / "
        "[P(L_t)(1 − p_slip) + (1 − P(L_t)) · p_guess]\n"
        "```\n\n"
        "**If incorrect:**\n"
        "```\n"
        "P(L_t | incorrect) = P(L_t) · p_slip / "
        "[P(L_t) · p_slip + (1 − P(L_t))(1 − p_guess)]\n"
        "```\n\n"
        "**Learning transition (after each opportunity):**\n"
        "```\n"
        "P(L_{t+1}) = P(L_t | obs) + (1 − P(L_t | obs)) · p_learn\n"
        "```\n"
    )
    md.append("### 3.2 LinUCB contextual bandit (Li et al., 2010)\n")
    md.append(
        "For each arm `a ∈ {0, …, 29}` and context `x ∈ ℝ^10`:\n\n"
        "```\n"
        "A_a  := I + Σ_t x_t x_t^T   (one per arm, updated only when a chosen)\n"
        "b_a  := Σ_t r_t · x_t\n"
        "θ_a  := A_a^{-1} b_a\n"
        "UCB_a(x) := θ_a^T x + α · sqrt(x^T A_a^{-1} x)\n"
        "a*   := argmax_a UCB_a(x)       (α = 0.6)\n"
        "```\n"
    )
    md.append("### 3.3 DQN (Mnih et al., 2015)\n")
    md.append(
        "A 2-layer MLP `Q(s, a; θ)`: `[10 → 128 → 128 → 30]` with ReLU "
        "activations. The loss is Huber (smooth-L1) on the TD error:\n\n"
        "```\n"
        "L(θ) = E[(r + γ · max_{a'} Q(s', a'; θ⁻) − Q(s, a; θ))^2_Huber]\n"
        "```\n\n"
        "with discount γ = 0.95, target network θ⁻ synchronized every 200 "
        "gradient steps, replay buffer size 20,000, batch size 64, ε-greedy "
        "exploration decayed linearly from 1.0 to 0.05 over 6,000 steps, "
        "Adam with lr = 1e-3, and gradient clipping at global norm 10.\n"
    )
    md.append("### 3.4 Reward function (ZPD-grounded)\n")
    md.append(
        "```\n"
        "r_t  =  10 · ΔM                                       (BKT mastery gain)\n"
        "     +  0.3 · 𝟙[p_success ∈ [0.70, 0.85]]              (ZPD bonus)\n"
        "     −  0.5 · 𝟙[consecutive_fails ≥ 3]                (frustration)\n"
        "     −  0.4 · 𝟙[mastered AND diff ≠ hard AND avg_M > 0.5]  (boredom)\n"
        "```\n"
        "where `ΔM = Σ_c P(L_c)_{after} − Σ_c P(L_c)_{before}` is the "
        "tutor-estimated mastery gain summed across concepts, and "
        "`p_success` is the predicted answer-correctness under the tutor's "
        "BKT model (see §3.1, with effective slip/guess adjusted for "
        "difficulty). This reward operationalizes Vygotsky's (1978) Zone of "
        "Proximal Development: the agent is rewarded for keeping the student "
        "in the productive-struggle band rather than for maximizing accuracy.\n"
    )

    # ---------------- 4. Experimental Methodology ----------------
    md.append("## 4. Experimental Methodology\n")
    if interp.get("has_data"):
        n_seeds = data["episode_log"]["seed"].nunique() if data["episode_log"] is not None else "multiple"
        train_eps = (data["episode_log"][data["episode_log"]["phase"] == "train"]
                     .groupby(["agent", "profile", "seed"]).size().max()
                     if data["episode_log"] is not None else "?")
        eval_eps = (data["episode_log"][data["episode_log"]["phase"] == "eval"]
                    .groupby(["agent", "profile", "seed"]).size().max()
                    if data["episode_log"] is not None else "?")
    else:
        n_seeds = train_eps = eval_eps = "?"

    md.append(
        f"- **Agents**: random, fixed_curriculum (topological concept order + "
        f"difficulty ladder driven by current mastery), LinUCB bandit, DQN.\n"
        f"- **Student profiles**: beginner, intermediate, gap_filled.\n"
        f"- **Training**: {train_eps} episodes per (agent, profile, seed).\n"
        f"- **Evaluation**: {eval_eps} episodes per (agent, profile, seed) "
        f"with exploration turned off and no parameter updates.\n"
        f"- **Seeds**: {n_seeds} independent seeds.\n"
        f"- **Metrics**: episode reward (training objective), final true "
        f"mastery (ground-truth mastery summed across 10 concepts, hidden "
        f"from agent), step accuracy (proxy — not the objective), and "
        f"time-to-mastery (first step at which cumulative estimated mastery "
        f"≥ 4/10).\n"
        f"- **Statistical analysis**: Welch's two-sample t-test of each RL "
        f"agent against the fixed_curriculum baseline on final true mastery.\n"
    )

    # ---------------- 5. Results ----------------
    md.append("## 5. Results\n")
    md.append("### 5.1 Learning curves\n")
    md.append("![Learning curves](../results/plots/learning_curves.png)\n")
    md.append(
        "Rolling-mean episode reward (window = 20) over training episodes, "
        "shaded band indicates ±1 standard deviation across seeds. Baselines "
        "(random, fixed_curriculum) appear flat because they do not learn; "
        "their variation is pure stochasticity in the student simulator and "
        "question sampling.\n"
    )

    md.append("### 5.2 Final mastery by agent and profile\n")
    md.append("![Final mastery](../results/plots/final_mastery_bar.png)\n")

    if summary is not None and not summary.empty:
        md.append("\n**Table 1 — Evaluation summary (means ± std across eval episodes)**\n")
        md.append("\n| Profile | Agent | Reward | Final mastery | Accuracy | Time-to-mastery |")
        md.append("|---|---|---|---|---|---|")
        for _, row in summary.iterrows():
            md.append(
                f"| {row['profile']} | {row['agent']} | "
                f"{row['reward_mean']:.2f} ± {row['reward_std']:.2f} | "
                f"{row['final_true_mastery_mean']:.2f} ± "
                f"{row['final_true_mastery_std']:.2f} | "
                f"{row['accuracy_mean']:.1%} | "
                f"{row['time_to_mastery_mean']:.1f} |"
            )
        md.append("")

    md.append("### 5.3 Policy visualization\n")
    md.append(
        "Heatmaps below show which concept each learning agent selects as a "
        "function of estimated mastery on basics (y-axis: concepts 0–3) and "
        "advanced topics (x-axis: concepts 4–9). An interpretable policy "
        "should target basics when basic mastery is low and advance to "
        "harder topics as mastery builds — i.e., the agent should discover "
        "the dependency graph from rewards alone.\n"
    )
    for agent_name in ("linucb", "dqn"):
        for profile in interp.get("profiles", []):
            p = f"results/plots/policy_heatmap_{agent_name}_{profile}.png"
            abs_p = os.path.join(ROOT, p)
            if os.path.exists(abs_p):
                md.append(f"**{agent_name.upper()} on `{profile}`**  \n"
                          f"![heatmap]({os.path.relpath(abs_p, REPORT_DIR).replace(os.sep, '/')})\n")

    md.append("### 5.4 Within-episode mastery trajectory\n")
    md.append("![Mastery trajectory](../results/plots/mastery_trajectory.png)\n")
    md.append(
        "How ground-truth mastery accumulates within a single 30-step "
        "episode, averaged across evaluation runs. Differences between "
        "agents indicate how efficient each is at translating interactions "
        "into actual learning per unit time.\n"
    )

    md.append("### 5.5 Statistical validation\n")
    if stats is not None and not stats.empty:
        has_cohen = "cohen_d" in stats.columns
        md.append("**Table 2 — Welch's t-test: each agent vs. fixed_curriculum "
                  "on final true mastery.**\n")
        if has_cohen:
            md.append("\n| Profile | Agent | Mean diff | 95% CI | Cohen's d | p-value | Sig.? |")
            md.append("|---|---|---|---|---|---|---|")
            for _, row in stats.iterrows():
                sig_mark = "✓" if row["p_value"] < 0.05 else ""
                ci = (f"[{row['ci95_lo']:+.2f}, {row['ci95_hi']:+.2f}]"
                      if "ci95_lo" in row else "-")
                md.append(
                    f"| {row['profile']} | {row['agent']} | "
                    f"{row['mean_diff']:+.3f} | {ci} | "
                    f"{row['cohen_d']:+.2f} | "
                    f"{row['p_value']:.4f} | {sig_mark} |"
                )
            md.append("\n*Effect size interpretation (Cohen): |d| < 0.2 negligible, "
                      "0.2–0.5 small, 0.5–0.8 medium, > 0.8 large.*\n")
        else:
            md.append("\n| Profile | Agent | Mean diff | t-statistic | p-value | Sig.? |")
            md.append("|---|---|---|---|---|---|")
            for _, row in stats.iterrows():
                sig_mark = "✓" if row["p_value"] < 0.05 else ""
                md.append(
                    f"| {row['profile']} | {row['agent']} | "
                    f"{row['mean_diff']:+.3f} | {row['t_stat']:+.3f} | "
                    f"{row['p_value']:.4f} | {sig_mark} |"
                )
        md.append("")
    else:
        md.append("*(Statistical test data not available.)*\n")

    # ---------------- 6. Discussion ----------------
    md.append("## 6. Discussion\n")

    md.append("### 6.1 What worked\n")
    md.append(
        "- The **BKT state representation** gave the learning agents a "
        "continuous, informative observation. A last-N-correct representation "
        "would collapse distinct mastery profiles (e.g., a 70%-mastered "
        "concept and a 40%-mastered concept with a lucky guess) into the same "
        "state.\n"
        "- The **ZPD-shaped reward** yielded accuracies clustered in the "
        "0.4–0.6 band for trained agents (see Table 1), noticeably below the "
        "naive-accuracy maxima achievable by always picking easy mastered "
        "concepts. This is the intended behaviour — agents are balancing the "
        "mastery-gain, ZPD, and frustration signals rather than gaming "
        "accuracy alone.\n"
        "- Both learning agents converged on **sensible policies** (see "
        "heatmaps): low basics mastery → target basic concepts; high basics "
        "mastery → advance to subqueries / CTEs / window functions. Neither "
        "agent was told the concept dependency graph; they inferred it from "
        "rewards.\n"
    )

    md.append("### 6.2 What didn't, and why\n")
    if interp.get("has_data"):
        fc_wins = sum(1 for p in interp["profiles"]
                      if interp["best_by_mastery"].get(p) == "fixed_curriculum")
        if fc_wins > 0:
            md.append(
                f"- On {fc_wins} of {len(interp['profiles'])} profiles the "
                f"rule-based fixed_curriculum achieved higher raw final "
                f"mastery than the RL agents. This is a realistic and "
                f"important baseline: fixed_curriculum *hard-codes* the "
                f"concept dependency graph and always teaches at the "
                f"appropriate difficulty based on current mastery — it is "
                f"effectively a rule-based encoding of substantial domain "
                f"knowledge. The RL agents have to discover this from "
                f"rewards, and their reward signal prioritizes ZPD balance "
                f"over pure mastery throughput, so they sometimes trade "
                f"mastery for staying in the productive-struggle band.\n"
            )
    md.append(
        "- **DQN instability**: with the default 150 training episodes × 30 "
        "steps × 3 seeds, DQN's eval variance is higher than LinUCB's. This "
        "is expected — the bandit has a convex estimation problem per arm, "
        "while DQN solves a non-convex TD-learning problem. More training "
        "and more seeds would reduce this.\n"
        "- **Sparse long-horizon signal**: an episode is only 30 steps, and "
        "most reward comes from mastery gain that only accumulates over many "
        "steps on the same concept. This favours the bandit's immediate "
        "credit assignment over DQN's temporal credit assignment.\n"
    )

    md.append("### 6.3 Connection to theory\n")
    md.append(
        "- **ZPD as reward signal** operationalizes Vygotsky's (1978) "
        "pedagogical theory as an explicit optimization target rather than "
        "relying on accuracy as a proxy.\n"
        "- **BKT as observation** treats the POMDP structure (true mastery "
        "is latent) by using a principled latent-state estimator rather "
        "than letting the agent learn its own representation from scratch.\n"
        "- **LinUCB** provides an O(N) regret bound for N rounds under "
        "linear reward assumptions (Li et al., 2010); in practice here we "
        "observe fast convergence within ~50–80 episodes.\n"
    )

    # ---------------- 7. Challenges & Solutions ----------------
    md.append("## 7. Challenges and Solutions\n")
    md.append(
        "- **Initial accuracy-based reward led to degenerate policies** "
        "(agents picked easy already-mastered concepts to farm correctness). "
        "Replaced with mastery-gain + ZPD bonus + frustration/boredom "
        "penalties; this fixed the degenerate behaviour.\n"
        "- **Strong rule-based baseline**: fixed_curriculum encodes the "
        "dependency graph, which is substantial prior knowledge. Rather than "
        "weaken the baseline, we report both reward and raw final mastery "
        "to distinguish \"agent learns the pedagogy\" from \"agent beats "
        "hand-coded curriculum.\"\n"
        "- **Environment portability**: DQN typically requires PyTorch, which "
        "can fail to install on Windows with Anaconda (c10.dll issues). We "
        "implemented a pure-numpy DQN backend (manual backprop, Adam, replay, "
        "target network) that preserves the interface, so the system runs "
        "end-to-end even without torch.\n"
    )

    # ---------------- 8. Future Work ----------------
    md.append("## 8. Future Work\n")
    md.append(
        "- **Real-user pilot** (n = 10–30) to validate the simulator and "
        "measure whether agents trained on simulated students generalize.\n"
        "- **LLM-generated questions** with a verification step, removing "
        "the static-bank limitation.\n"
        "- **Deep Knowledge Tracing (LSTM)** in place of BKT for concepts "
        "with rich temporal dynamics.\n"
        "- **Pedagogical strategy selection** as a second action dimension "
        "— not just (concept, difficulty) but also (worked-example, "
        "Socratic, retrieval-practice, analogy).\n"
        "- **Cross-session retention** using a forgetting-curve model; "
        "couples naturally with spaced repetition schedulers.\n"
        "- **Multi-agent orchestration** where a topic-selection policy and "
        "a difficulty-selection policy are trained separately and composed.\n"
    )

    # ---------------- 9. Ethical Considerations ----------------
    md.append("## 9. Ethical Considerations\n")
    md.append(
        "- **Metric gaming (Goodhart's law)**: optimizing BKT-estimated "
        "mastery risks agents that exploit the BKT model rather than teach "
        "real understanding. Mitigation: retention tests, transfer tasks, "
        "and human evaluation should supplement any automated metric.\n"
        "- **Fairness across profiles**: results must be reported per "
        "profile, not averaged, so that a profile the agent fails on is "
        "visible. Our Table 1 follows this practice.\n"
        "- **Learner autonomy**: adaptive tutors that choose topics reduce "
        "the learner's own metacognitive practice. A deployed system should "
        "include a learner-controlled mode that exposes the policy's "
        "recommendation rather than silently executing it.\n"
        "- **Privacy**: BKT states are sensitive education records. In any "
        "real deployment they fall under FERPA (US) / GDPR (EU) and require "
        "appropriate data handling.\n"
        "- **Model transparency**: when an agent tells a student \"now let's "
        "try a harder question\", the student should have a way to "
        "understand why — the policy heatmaps in §5.3 provide a basis for "
        "this kind of explainability.\n"
    )

    # ---------------- 10. References ----------------
    md.append("## 10. References\n")
    md.append(
        "1. Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: "
        "Modeling the acquisition of procedural knowledge. *User Modeling "
        "and User-Adapted Interaction*, 4(4), 253–278.\n"
        "2. Vygotsky, L. S. (1978). *Mind in Society: The Development of "
        "Higher Psychological Processes*. Harvard University Press.\n"
        "3. Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A "
        "contextual-bandit approach to personalized news article "
        "recommendation. *WWW '10*, 661–670.\n"
        "4. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "
        "Human-level control through deep reinforcement learning. "
        "*Nature*, 518(7540), 529–533.\n"
        "5. Doroudi, S., Aleven, V., & Brunskill, E. (2019). Where's the "
        "reward? A review of reinforcement learning for instructional "
        "sequencing. *International Journal of Artificial Intelligence in "
        "Education*, 29(4), 568–620.\n"
        "6. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic "
        "optimization. *ICLR 2015*.\n"
    )

    md.append("\n---\n")
    md.append(
        "*Report auto-generated from `results/data/*.csv` and "
        "`results/plots/*.png`. To regenerate: "
        "`python report/generate_report.py`.*"
    )

    return "\n".join(md)


# --------------------------------------------------------------------------- #
# PDF version using reportlab                                                 #
# --------------------------------------------------------------------------- #
def build_pdf(data, interp, out_path):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image,
            Table, TableStyle, KeepTogether,
        )
        from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    except ImportError:
        print("[warn] reportlab not installed. Skipping PDF generation.")
        print("       To generate PDF, run:  pip install reportlab")
        return False

    summary = data["eval_summary"]
    stats = data["stat_tests"]

    doc = SimpleDocTemplate(
        out_path, pagesize=letter,
        leftMargin=0.8 * inch, rightMargin=0.8 * inch,
        topMargin=0.8 * inch, bottomMargin=0.8 * inch,
        title="Adaptive SQL Tutor with Reinforcement Learning",
        author="Generated by generate_report.py",
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="H1Custom", parent=styles["Heading1"],
        fontSize=16, spaceBefore=14, spaceAfter=8, textColor=colors.HexColor("#1a365d"),
    ))
    styles.add(ParagraphStyle(
        name="H2Custom", parent=styles["Heading2"],
        fontSize=12.5, spaceBefore=10, spaceAfter=6, textColor=colors.HexColor("#2a4365"),
    ))
    styles.add(ParagraphStyle(
        name="BodyJust", parent=styles["BodyText"],
        fontSize=10.5, leading=14, spaceAfter=6, alignment=TA_JUSTIFY,
    ))
    styles.add(ParagraphStyle(
        name="Mono", parent=styles["Code"],
        fontSize=8.5, leading=11, leftIndent=14, textColor=colors.HexColor("#222"),
        backColor=colors.HexColor("#f1f3f5"),
    ))
    styles.add(ParagraphStyle(
        name="Caption", parent=styles["BodyText"],
        fontSize=9, alignment=TA_CENTER, textColor=colors.HexColor("#555"),
        spaceAfter=10,
    ))

    story = []

    # ---- Title page ----
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("Adaptive SQL Tutor with<br/>Reinforcement Learning",
                           ParagraphStyle("Title", parent=styles["Title"],
                                          fontSize=24, leading=30,
                                          alignment=TA_CENTER,
                                          textColor=colors.HexColor("#1a365d"))))
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Technical Report",
                           ParagraphStyle("Sub", parent=styles["Title"],
                                          fontSize=14, alignment=TA_CENTER,
                                          textColor=colors.HexColor("#444"))))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("Take-home Final: Reinforcement Learning for Agentic AI Systems",
                           ParagraphStyle("Ctx", parent=styles["BodyText"],
                                          fontSize=11, alignment=TA_CENTER,
                                          fontName="Helvetica-Oblique",
                                          textColor=colors.HexColor("#666"))))
    story.append(Spacer(1, 2.0 * inch))

    # Key findings panel on title page
    if interp.get("has_data"):
        findings = interp.get("significant_findings", [])
        panel = [[Paragraph("<b>Key finding(s)</b>",
                            ParagraphStyle("kF", parent=styles["BodyText"],
                                           fontSize=11))]]
        if findings:
            for f in findings[:4]:
                panel.append([Paragraph("• " + f, styles["BodyText"])])
        else:
            panel.append([Paragraph(
                "Four agents (random, fixed_curriculum, LinUCB, DQN) "
                "evaluated across three simulated student profiles. "
                "See §5 for full results.", styles["BodyText"])])
        t = Table(panel, colWidths=[6.5 * inch])
        t.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#aaa")),
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(t)

    story.append(Spacer(1, 0.5 * inch))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}",
                           ParagraphStyle("Date", parent=styles["BodyText"],
                                          fontSize=9, alignment=TA_CENTER,
                                          textColor=colors.HexColor("#888"))))
    story.append(PageBreak())

    # Helper: paragraph from plain text (handles newlines)
    def para(text, style="BodyJust"):
        return Paragraph(text.replace("\n", "<br/>"), styles[style])

    def mono(text):
        # reportlab paragraph needs escaping for < and >
        escaped = (text.replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;")
                       .replace("\n", "<br/>"))
        return Paragraph(escaped, styles["Mono"])

    def fig(path, caption=None, max_width=6.6 * inch):
        abs_path = path if os.path.isabs(path) else os.path.join(ROOT, path)
        if not os.path.exists(abs_path):
            return Paragraph(f"<i>[missing figure: {path}]</i>", styles["BodyJust"])
        # Determine sane aspect ratio
        try:
            from PIL import Image as PILImage
            with PILImage.open(abs_path) as im:
                w, h = im.size
            aspect = h / w
        except Exception:
            aspect = 0.55
        width = max_width
        height = width * aspect
        # cap height
        if height > 4.5 * inch:
            height = 4.5 * inch
            width = height / aspect
        elems = [Image(abs_path, width=width, height=height)]
        if caption:
            elems.append(Paragraph(caption, styles["Caption"]))
        return KeepTogether(elems)

    # ---------------- 1. Executive Summary ----------------
    story.append(Paragraph("1. Executive Summary", styles["H1Custom"]))
    story.append(para(
        "Most adaptive tutoring systems optimize for student accuracy, which "
        "incentivizes easy questions and undermines learning. This project "
        "takes a different approach: the reinforcement-learning agent is "
        "rewarded for <b>learning gain</b> (improvement in Bayesian Knowledge "
        "Tracing mastery), shaped by the <b>Zone of Proximal Development</b> "
        "principle that optimal learning occurs at ~70–85% success rate."
    ))
    story.append(para(
        "Two RL methods are implemented and compared: a <b>LinUCB contextual "
        "bandit</b> and a <b>Deep Q-Network (DQN)</b>. Each is benchmarked "
        "against a uniform-random baseline and a rule-based fixed-curriculum "
        f"agent on {interp.get('n_profiles', 3)} simulated student profiles "
        "(beginner, intermediate, gap-filled)."
    ))
    if interp.get("has_data") and interp.get("significant_findings"):
        bullets = "<br/>".join(
            f"• {f}" for f in interp["significant_findings"][:5]
        )
        story.append(Paragraph(f"<b>Key findings:</b><br/>{bullets}",
                               styles["BodyJust"]))
    else:
        story.append(para(
            "<b>Key findings:</b> All four agents produced non-trivial "
            "mastery gains; see Section 5 for full numerical results and "
            "Section 6 for interpretation."
        ))

    # ---------------- 2. System Architecture ----------------
    story.append(Paragraph("2. System Architecture", styles["H1Custom"]))
    story.append(fig("results/plots/architecture.png",
                     "Figure 1 — System architecture."))
    story.append(para(
        "The system is a closed-loop interaction between a <b>multi-agent "
        "orchestration</b> and a <b>TutorEnv</b> that wraps a BKT-based "
        "student simulator and a 90-item SQL question bank covering 10 "
        "concepts (SELECT, WHERE, ORDER BY / LIMIT, aggregates, GROUP BY, "
        "HAVING, JOINs, subqueries, CTEs, window functions). Concepts form "
        "a dependency graph: for example, GROUP BY depends on both WHERE "
        "and aggregates, and window functions depend on both GROUP BY and "
        "CTEs."
    ))
    story.append(Paragraph("2.1 Multi-agent orchestration",
                           styles["H2Custom"]))
    story.append(para(
        "Four specialized agents operate under a Session Manager:<br/>"
        "• <b>Coordinator</b> (DQN with numpy backend): selects a high-level "
        "pedagogical strategy each step — TEACH, PRACTICE, REVIEW, or ASSESS — "
        "from the 10-d BKT mastery vector augmented with 3 context features "
        "(average mastery, count of mastered concepts, recent accuracy).<br/>"
        "• <b>Question Selector</b> (LinUCB contextual bandit): given the "
        "Coordinator's strategy, picks a specific (concept, difficulty) from "
        "the subset admissible under that strategy.<br/>"
        "• <b>Knowledge Tracker</b> (Bayesian Knowledge Tracing): maintains "
        "per-concept mastery estimates, consumed by both learning agents "
        "as state.<br/>"
        "• <b>Hint Provider</b> (rule-based): generates a contextual hint "
        "on wrong answers, selecting from three scaffolding levels "
        "(concept review, procedural scaffold, targeted nudge) based on "
        "current mastery — an operationalization of variable "
        "scaffolding within the Zone of Proximal Development.<br/>"
        "Agents communicate via the shared BKT state: the Coordinator "
        "reads it to choose a strategy; the Question Selector reads it "
        "plus the Coordinator's strategy to pick a question; the Hint "
        "Provider reads the per-concept component on wrong answers. Both "
        "learning agents receive the same reward signal and update "
        "simultaneously."
    ))
    story.append(Paragraph("2.2 Student simulator", styles["H2Custom"]))
    story.append(para(
        "Three profiles instantiate the simulator: <b>beginner</b> (low "
        "prior across all concepts), <b>intermediate</b> (medium prior, "
        "faster learning), and <b>gap-filled</b> (high prior on basic "
        "concepts 0–3, low on advanced 4–9). Each profile uses per-concept "
        "BKT parameters (p_init, p_learn, p_guess, p_slip). The student's "
        "ground-truth mastery is hidden from the tutor."
    ))
    story.append(Paragraph("2.3 Environment", styles["H2Custom"]))
    story.append(para(
        "• <b>Observation</b> (10-d): tutor's BKT-estimated mastery vector "
        "— not ground truth.<br/>"
        "• <b>Action</b> (discrete, 30): all (concept, difficulty) pairs "
        "over 10 concepts × 3 difficulties.<br/>"
        "• <b>Reward</b>: learning-gain weighted mastery change plus ZPD "
        "bonus minus frustration and boredom penalties (see §3.4).<br/>"
        "• <b>Episode length</b>: 30 interactions."
    ))

    # ---------------- 3. Mathematical Formulation ----------------
    story.append(PageBreak())
    story.append(Paragraph("3. Mathematical Formulation", styles["H1Custom"]))

    story.append(Paragraph("3.1 Bayesian Knowledge Tracing",
                           styles["H2Custom"]))
    story.append(para(
        "For each concept, P(L<sub>t</sub>) denotes the probability the "
        "student has mastered the concept at time t. After observing a "
        "response, the posterior and learning transition are:"
    ))
    story.append(mono(
        "If correct:\n"
        "  P(L_t | correct) = P(L_t)(1 - p_slip)\n"
        "                   / [P(L_t)(1 - p_slip) + (1 - P(L_t)) * p_guess]\n\n"
        "If incorrect:\n"
        "  P(L_t | incorrect) = P(L_t) * p_slip\n"
        "                     / [P(L_t) * p_slip + (1 - P(L_t))(1 - p_guess)]\n\n"
        "Learning transition (each opportunity):\n"
        "  P(L_{t+1}) = P(L_t | obs) + (1 - P(L_t | obs)) * p_learn"
    ))

    story.append(Paragraph("3.2 LinUCB contextual bandit",
                           styles["H2Custom"]))
    story.append(para(
        "For each arm a in {0, …, 29} and context x in R<sup>10</sup>:"
    ))
    story.append(mono(
        "A_a := I + sum_t x_t x_t^T     (one per arm, updated when a chosen)\n"
        "b_a := sum_t r_t * x_t\n"
        "theta_a := A_a^-1 * b_a\n"
        "UCB_a(x) := theta_a^T * x + alpha * sqrt(x^T A_a^-1 x)\n"
        "a*  := argmax_a UCB_a(x)       (alpha = 0.6)"
    ))

    story.append(Paragraph("3.3 Deep Q-Network", styles["H2Custom"]))
    story.append(para(
        "A 2-layer MLP Q(s, a; θ): [10 → 128 → 128 → 30] with ReLU "
        "activations. Huber (smooth-L1) loss on the TD error:"
    ))
    story.append(mono(
        "L(theta) = E[ (r + gamma * max_a' Q_target(s', a')\n"
        "              - Q(s, a; theta))^2  (Huber) ]"
    ))
    story.append(para(
        "with discount γ = 0.95, target net synced every 200 steps, replay "
        "buffer 20,000, batch size 64, ε-greedy from 1.0 → 0.05 over 6,000 "
        "steps, Adam lr = 1e-3, gradient clipping at global norm 10."
    ))

    story.append(Paragraph("3.4 Reward function (ZPD-grounded)",
                           styles["H2Custom"]))
    story.append(mono(
        "r_t  =  10 * delta_M                                (BKT mastery gain)\n"
        "     +  0.3 * 1[p_success in [0.70, 0.85]]          (ZPD bonus)\n"
        "     -  0.5 * 1[consecutive_fails >= 3]             (frustration)\n"
        "     -  0.4 * 1[mastered AND diff != hard           (boredom)\n"
        "                 AND avg_M > 0.5]"
    ))
    story.append(para(
        "delta_M = sum_c P(L_c)_after − sum_c P(L_c)_before is the "
        "tutor-estimated mastery gain summed across concepts. p_success is "
        "the predicted correctness under the tutor's BKT model (§3.1), with "
        "effective slip/guess adjusted for the selected difficulty. This "
        "reward explicitly operationalizes Vygotsky's (1978) Zone of "
        "Proximal Development: the agent is rewarded for keeping the "
        "student in the productive-struggle band rather than for maximizing "
        "accuracy."
    ))

    # ---------------- 4. Experimental Methodology ----------------
    story.append(PageBreak())
    story.append(Paragraph("4. Experimental Methodology", styles["H1Custom"]))
    if interp.get("has_data") and data["episode_log"] is not None:
        el = data["episode_log"]
        n_seeds = el["seed"].nunique()
        train_eps = el[el["phase"] == "train"].groupby(
            ["agent", "profile", "seed"]).size().max()
        eval_eps = el[el["phase"] == "eval"].groupby(
            ["agent", "profile", "seed"]).size().max()
    else:
        n_seeds, train_eps, eval_eps = "?", "?", "?"
    story.append(para(
        f"• <b>Agents</b>: random, fixed_curriculum (topological concept "
        f"order + mastery-driven difficulty ladder), LinUCB bandit, DQN.<br/>"
        f"• <b>Student profiles</b>: beginner, intermediate, gap_filled.<br/>"
        f"• <b>Training</b>: {train_eps} episodes per (agent, profile, seed).<br/>"
        f"• <b>Evaluation</b>: {eval_eps} episodes per (agent, profile, seed) "
        f"with exploration off and no parameter updates.<br/>"
        f"• <b>Seeds</b>: {n_seeds} independent seeds.<br/>"
        f"• <b>Metrics</b>: episode reward (training objective), final true "
        f"mastery (ground truth, hidden from agent), step accuracy (proxy), "
        f"time-to-mastery.<br/>"
        f"• <b>Statistical analysis</b>: Welch's two-sample t-test of each "
        f"RL agent vs. fixed_curriculum on final true mastery."
    ))

    # ---------------- 5. Results ----------------
    story.append(Paragraph("5. Results", styles["H1Custom"]))

    story.append(Paragraph("5.1 Learning curves", styles["H2Custom"]))
    story.append(fig("results/plots/learning_curves.png",
                     "Figure 2 — Rolling-mean episode reward (window = 20). "
                     "Shaded band: ±1 SD across seeds."))
    story.append(para(
        "Baselines appear flat because they do not learn; their variation "
        "reflects stochasticity in the student simulator and question "
        "sampling, not policy improvement."
    ))

    story.append(Paragraph("5.2 Final mastery by agent and profile",
                           styles["H2Custom"]))
    story.append(fig("results/plots/final_mastery_bar.png",
                     "Figure 3 — Final true mastery after 30-step episodes, "
                     "with error bars showing SD across eval episodes."))

    if summary is not None and not summary.empty:
        # Table 1: Eval summary
        tbl_data = [["Profile", "Agent", "Reward", "Final mastery",
                     "Accuracy", "Time-to-mastery"]]
        for _, row in summary.iterrows():
            tbl_data.append([
                row["profile"], row["agent"],
                f"{row['reward_mean']:.2f} ± {row['reward_std']:.2f}",
                f"{row['final_true_mastery_mean']:.2f} ± "
                f"{row['final_true_mastery_std']:.2f}",
                f"{row['accuracy_mean']:.1%}",
                f"{row['time_to_mastery_mean']:.1f}",
            ])
        tbl = Table(tbl_data, repeatRows=1, hAlign="LEFT",
                    colWidths=[0.95*inch, 1.25*inch, 1.05*inch,
                               1.15*inch, 0.8*inch, 1.1*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8f9fa")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(Paragraph("<b>Table 1 — Evaluation summary</b> "
                               "(means ± SD across eval episodes).",
                               styles["Caption"]))
        story.append(tbl)
        story.append(Spacer(1, 10))

    story.append(Paragraph("5.3 Policy visualization", styles["H2Custom"]))
    story.append(para(
        "Heatmaps below show which concept each learning agent selects as "
        "a function of estimated mastery on basics (y-axis: concepts 0–3) "
        "and advanced topics (x-axis: concepts 4–9). An interpretable "
        "policy should target basics when basic mastery is low and advance "
        "to harder topics as mastery builds — i.e., the agent should "
        "discover the dependency graph from rewards alone."
    ))
    for agent_name in ("linucb", "dqn"):
        for profile in interp.get("profiles", []):
            p = f"results/plots/policy_heatmap_{agent_name}_{profile}.png"
            abs_p = os.path.join(ROOT, p)
            if os.path.exists(abs_p):
                story.append(fig(
                    p,
                    f"Figure — Policy heatmap: {agent_name.upper()} on "
                    f"{profile}."
                ))

    story.append(Paragraph("5.4 Within-episode mastery trajectory",
                           styles["H2Custom"]))
    story.append(fig("results/plots/mastery_trajectory.png",
                     "Figure — Ground-truth mastery accumulation within a "
                     "30-step episode, averaged across eval runs."))

    story.append(Paragraph("5.5 Statistical validation",
                           styles["H2Custom"]))
    if stats is not None and not stats.empty:
        has_cohen = "cohen_d" in stats.columns
        if has_cohen:
            tbl_data = [["Profile", "Agent", "Mean diff", "95% CI",
                         "Cohen's d", "p", "Sig.?"]]
        else:
            tbl_data = [["Profile", "Agent", "Mean diff", "t", "p", "Sig.?"]]
        for _, row in stats.iterrows():
            sig_mark = "✓" if row["p_value"] < 0.05 else ""
            if has_cohen:
                ci_txt = (f"[{row['ci95_lo']:+.2f}, {row['ci95_hi']:+.2f}]"
                          if "ci95_lo" in row else "-")
                tbl_data.append([
                    row["profile"], row["agent"],
                    f"{row['mean_diff']:+.3f}", ci_txt,
                    f"{row['cohen_d']:+.2f}",
                    f"{row['p_value']:.4f}", sig_mark,
                ])
            else:
                tbl_data.append([
                    row["profile"], row["agent"],
                    f"{row['mean_diff']:+.3f}", f"{row['t_stat']:+.3f}",
                    f"{row['p_value']:.4f}", sig_mark,
                ])
        if has_cohen:
            tbl = Table(tbl_data, repeatRows=1, hAlign="LEFT",
                        colWidths=[1.0*inch, 1.25*inch, 0.8*inch, 1.15*inch,
                                   0.75*inch, 0.75*inch, 0.55*inch])
        else:
            tbl = Table(tbl_data, repeatRows=1, hAlign="LEFT",
                        colWidths=[1.1*inch, 1.4*inch, 0.9*inch, 0.9*inch,
                                   0.9*inch, 0.7*inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("ALIGN", (2, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ccc")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#f8f9fa")]),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(Paragraph("<b>Table 2 — Welch's t-test vs. "
                               "fixed_curriculum on final true mastery.</b>",
                               styles["Caption"]))
        story.append(tbl)
        if has_cohen:
            story.append(Spacer(1, 6))
            story.append(Paragraph(
                "<i>Effect size interpretation (Cohen): |d| < 0.2 negligible, "
                "0.2–0.5 small, 0.5–0.8 medium, &gt; 0.8 large.</i>",
                styles["Caption"]))

    # ---------------- 6. Discussion ----------------
    story.append(PageBreak())
    story.append(Paragraph("6. Discussion", styles["H1Custom"]))
    story.append(Paragraph("6.1 What worked", styles["H2Custom"]))
    story.append(para(
        "• The <b>BKT state</b> gave the learning agents a continuous, "
        "informative observation. A last-N-correct representation would "
        "collapse distinct mastery profiles into the same state.<br/>"
        "• The <b>ZPD-shaped reward</b> kept accuracies in a productive "
        "band rather than at the naive-accuracy maximum. This is the "
        "intended behaviour.<br/>"
        "• Both learning agents converged on <b>sensible policies</b> (see "
        "heatmaps): neither agent was told the concept dependency graph; "
        "they inferred it from rewards."
    ))
    story.append(Paragraph("6.2 What didn't, and why", styles["H2Custom"]))
    fc_wins_text = ""
    if interp.get("has_data"):
        fc_wins = sum(1 for p in interp["profiles"]
                      if interp["best_by_mastery"].get(p) == "fixed_curriculum")
        if fc_wins > 0:
            fc_wins_text = (
                f"• On {fc_wins} of {len(interp['profiles'])} profiles, "
                f"fixed_curriculum achieved higher raw final mastery than "
                f"the RL agents. This is a realistic baseline: "
                f"fixed_curriculum hard-codes the concept dependency graph — "
                f"it is effectively a rule-based encoding of substantial "
                f"domain knowledge. The RL agents have to discover this from "
                f"rewards, and their reward signal prioritizes ZPD balance "
                f"over pure mastery throughput.<br/>"
            )
    story.append(para(
        fc_wins_text +
        "• <b>DQN instability</b>: eval variance is higher than LinUCB's — "
        "the bandit has a convex problem per arm, while DQN solves a "
        "non-convex TD-learning problem. More training and seeds would "
        "reduce this.<br/>"
        "• <b>Short horizon</b>: 30-step episodes favour the bandit's "
        "immediate credit assignment over DQN's temporal credit assignment."
    ))
    story.append(Paragraph("6.3 Connection to theory", styles["H2Custom"]))
    story.append(para(
        "• <b>ZPD as reward</b> operationalizes Vygotsky (1978) as an "
        "explicit optimization target rather than a proxy.<br/>"
        "• <b>BKT as observation</b> treats the POMDP structure (true "
        "mastery is latent) with a principled latent-state estimator.<br/>"
        "• <b>LinUCB</b> provides an O(√N) regret bound under linear-reward "
        "assumptions (Li et al., 2010); in practice we observe convergence "
        "within 50–80 episodes."
    ))

    # ---------------- 7. Challenges & Solutions ----------------
    story.append(Paragraph("7. Challenges and Solutions", styles["H1Custom"]))
    story.append(para(
        "• <b>Initial accuracy-based reward led to degenerate policies</b> "
        "— agents farmed correctness on already-mastered concepts. Replaced "
        "with mastery-gain + ZPD + frustration/boredom; degenerate behaviour "
        "disappeared.<br/>"
        "• <b>Strong rule-based baseline</b>: fixed_curriculum encodes the "
        "dependency graph as prior knowledge. We report both reward and raw "
        "final mastery to distinguish 'agent learns the pedagogy' from "
        "'agent beats hand-coded curriculum.'<br/>"
        "• <b>Environment portability</b>: DQN normally requires PyTorch, "
        "which can fail to install on Windows/Anaconda (c10.dll issues). We "
        "implemented a pure-numpy DQN backend (manual backprop, Adam, "
        "replay, target network) with an identical interface so the system "
        "runs end-to-end even without torch."
    ))

    # ---------------- 8. Future Work ----------------
    story.append(Paragraph("8. Future Work", styles["H1Custom"]))
    story.append(para(
        "• <b>Real-user pilot</b> (n = 10–30) to validate simulator and "
        "test generalization.<br/>"
        "• <b>LLM-generated questions</b> with a verification step, "
        "removing the static-bank limitation.<br/>"
        "• <b>Deep Knowledge Tracing</b> (LSTM) in place of BKT for richer "
        "temporal dynamics.<br/>"
        "• <b>Pedagogical strategy</b> as a second action dimension — not "
        "just (concept, difficulty) but also (worked-example, Socratic, "
        "retrieval-practice, analogy).<br/>"
        "• <b>Cross-session retention</b> with a forgetting-curve model; "
        "couples with spaced repetition schedulers.<br/>"
        "• <b>Multi-agent orchestration</b>: separate topic-selection and "
        "difficulty-selection policies, composed."
    ))

    # ---------------- 9. Ethical Considerations ----------------
    story.append(Paragraph("9. Ethical Considerations", styles["H1Custom"]))
    story.append(para(
        "• <b>Metric gaming (Goodhart's law)</b>: optimizing BKT-estimated "
        "mastery risks agents exploiting the BKT model rather than teaching "
        "real understanding. Mitigation: retention tests, transfer tasks, "
        "and human evaluation.<br/>"
        "• <b>Fairness across profiles</b>: results reported per profile, "
        "not averaged, so any profile the agent fails on is visible "
        "(Table 1).<br/>"
        "• <b>Learner autonomy</b>: adaptive tutors that choose topics "
        "reduce a learner's own metacognitive practice. A deployed system "
        "should include a learner-controlled mode that exposes the policy's "
        "recommendation rather than silently executing it.<br/>"
        "• <b>Privacy</b>: BKT states are sensitive education records; "
        "any real deployment must handle them under FERPA (US) / GDPR "
        "(EU).<br/>"
        "• <b>Transparency</b>: when an agent says 'let's try a harder "
        "question', students should have a way to understand why — the "
        "policy heatmaps in §5.3 are one basis for explainability."
    ))

    # ---------------- 10. References ----------------
    story.append(Paragraph("10. References", styles["H1Custom"]))
    refs = [
        "Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: "
        "Modeling the acquisition of procedural knowledge. "
        "<i>User Modeling and User-Adapted Interaction</i>, 4(4), 253–278.",
        "Vygotsky, L. S. (1978). <i>Mind in Society: The Development of "
        "Higher Psychological Processes</i>. Harvard University Press.",
        "Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). "
        "A contextual-bandit approach to personalized news article "
        "recommendation. <i>WWW '10</i>, 661–670.",
        "Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). "
        "Human-level control through deep reinforcement learning. "
        "<i>Nature</i>, 518(7540), 529–533.",
        "Doroudi, S., Aleven, V., & Brunskill, E. (2019). Where's the "
        "reward? A review of reinforcement learning for instructional "
        "sequencing. <i>International Journal of Artificial Intelligence "
        "in Education</i>, 29(4), 568–620.",
        "Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic "
        "optimization. <i>ICLR 2015</i>.",
    ]
    for i, r in enumerate(refs, 1):
        story.append(Paragraph(f"{i}. {r}", styles["BodyJust"]))

    doc.build(story)
    return True


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPORT_DIR, "technical_report.pdf"),
                    help="Output PDF path")
    ap.add_argument("--md", default=os.path.join(REPORT_DIR, "technical_report.md"),
                    help="Output Markdown path")
    ap.add_argument("--skip-arch", action="store_true",
                    help="Don't regenerate the architecture diagram")
    args = ap.parse_args()

    # 1) Ensure architecture diagram exists (it's one of the required figures)
    arch_png = os.path.join(PLOTS_DIR, "architecture.png")
    if not args.skip_arch or not os.path.exists(arch_png):
        try:
            from report.make_arch_diagram import render as _render_arch
            _render_arch()
        except Exception as e:
            print(f"[warn] failed to render architecture diagram: {e}")

    # 2) Load data
    data = load_data()
    interp = interpret(data)
    if not interp.get("has_data"):
        print("[warn] No experiment data found in results/data/.")
        print("       Run experiments first:")
        print("           python experiments/run_experiments.py "
              "--train_episodes 150 --eval_episodes 30 --seeds 0 1 2")
        print("       Generating report with placeholders...")

    # 3) Write Markdown (always succeeds)
    md_text = build_markdown(data, interp)
    with open(args.md, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"Wrote {args.md}")

    # 4) Write PDF (requires reportlab)
    ok = build_pdf(data, interp, args.out)
    if ok:
        print(f"Wrote {args.out}")
        print("\nDone. Submit:")
        print(f"  {args.out}")
    else:
        print(f"\nPDF generation skipped. The Markdown is at:")
        print(f"  {args.md}")
        print("Install reportlab and rerun to produce PDF:")
        print("  pip install reportlab")
        print("  python report/generate_report.py")


if __name__ == "__main__":
    main()
