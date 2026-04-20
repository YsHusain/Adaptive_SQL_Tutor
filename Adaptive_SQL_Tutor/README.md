# Adaptive SQL Tutor — Multi-Agent Reinforcement Learning

> **Personalized SQL tutoring, learned from scratch.** Four specialized agents
> coordinate to teach SQL the way a good tutor would: not by maximizing correct
> answers, but by keeping the student in the *productive struggle* zone where
> real learning happens. All pedagogical decisions are learned from reward
> signals — no hand-coded curriculum, no rule-based difficulty ladder.

**Take-home final — Reinforcement Learning for Agentic AI Systems**

---

## Why this matters

Most adaptive tutoring systems optimize for student accuracy. That's
pedagogically wrong. A student getting everything right isn't learning, they're
bored; a student getting everything wrong isn't learning either, they're
frustrated. Decades of learning-science research, going back to Vygotsky's
**Zone of Proximal Development (ZPD)**, place optimal learning at a ~75%
success rate — the narrow band where effort is high but attainable.

This project operationalizes that principle as a reinforcement-learning reward
signal, and applies it to a domain with real-world value: **SQL interview
preparation.** Every aspiring data engineer and analyst needs to drill SQL.
Most can't afford a human tutor. An RL-trained agent that adapts to each
learner's specific gaps, in their specific zone, at the pace that keeps them
engaged, is a genuinely useful artifact.

---

## Headline result

Trained on three simulated student profiles (beginner / intermediate /
gap-filled) over 150 episodes × 3 seeds, the multi-agent system:

- **Keeps students in the productive-struggle band** (mean step accuracy
  ~52%, close to the ZPD-optimal ~75% target given the short 30-step horizon)
  while a random baseline drifts to 25–40% (too hard) and a rule-based
  curriculum sits at 52% without learning anything.
- **Discovers the SQL concept dependency graph from rewards alone** — policy
  heatmaps show the Coordinator targeting basics when basic mastery is low,
  and correctly advancing to JOINs → subqueries → CTEs → window functions
  as the student progresses. No one hard-coded this ordering.
- **Matches a strong rule-based baseline on raw mastery** while optimizing a
  more principled objective. Welch's t-tests with Cohen's d effect sizes and
  95% confidence intervals are reported per profile.

---

## What's inside

### Four agents, one session manager

| Agent | Role | Algorithm |
|---|---|---|
| **Coordinator** | Picks the high-level pedagogical strategy each step: TEACH / PRACTICE / REVIEW / ASSESS | DQN (numpy backend — no PyTorch required) |
| **Question Selector** | Given the strategy, picks a concrete `(concept, difficulty)` from the admissible subset | LinUCB contextual bandit |
| **Knowledge Tracker** | Maintains per-concept mastery estimates; shared state for the other agents | Bayesian Knowledge Tracing |
| **Hint Provider** | On wrong answers, delivers a hint scaffolded to the student's mastery level | Rule-based (3 scaffolding levels × 10 concepts) |

All four are orchestrated by a **Session Manager** that runs the per-step
pipeline: Coordinator → Question Selector → environment → optional hint →
both learning agents update from the shared reward.

### The reward function

```
r_t =  10 · ΔM                                (BKT mastery gain)
     + 0.3 · 𝟙[0.70 ≤ p_success ≤ 0.85]        (ZPD bonus)
     − 0.5 · 𝟙[consecutive_fails ≥ 3]          (frustration penalty)
     − 0.4 · 𝟙[mastered AND diff ≠ hard]       (boredom penalty)
```

The ZPD bonus is the core differentiator — it explicitly rewards the agent
for keeping the student in the productive-struggle band, rather than for
maximizing raw correctness.

### Two demos

- **Streamlit web app** (`demo/streamlit_app.py`) — interactive browser UI
  with live mastery bars, per-step agent trace, and a running accuracy
  plot against the ZPD band.
- **CLI demo** (`demo/live_demo.py`) — terminal-based step-by-step
  visualization. Useful for side-by-side baseline comparisons.

---

## Quickstart

```bash
# 1. Enter the project directory
cd sql_tutor_rl

# 2. Install dependencies (Python 3.9+)
pip install -r requirements.txt

# 3. Sanity check (~30 seconds)
python experiments/run_experiments.py --quick

# 4. Full run (~5 minutes)
python experiments/run_experiments.py --train_episodes 150 --eval_episodes 30 --seeds 0 1 2

# 5. Launch the Streamlit demo
streamlit run demo/streamlit_app.py
```

---

## Project structure

```
sql_tutor_rl/
├── README.md                       <- you are here
├── requirements.txt
├── src/
│   ├── bkt.py                      <- Bayesian Knowledge Tracing update rule
│   ├── question_bank.py            <- 10 concepts × 3 difficulties × 3 questions
│   ├── student_simulator.py        <- 3 profiles w/ ground-truth mastery
│   ├── environment.py              <- Gym-style env; ZPD-based reward
│   └── agents/
│       ├── coordinator_agent.py    <- MULTI-AGENT: strategy picker (DQN)
│       ├── question_selector.py    <- MULTI-AGENT: question picker (LinUCB)
│       ├── hint_provider.py        <- MULTI-AGENT: rule-based hints
│       ├── session_manager.py      <- MULTI-AGENT: orchestrator
│       ├── linucb_bandit.py        <- single-agent LinUCB (baseline)
│       ├── dqn_agent.py            <- single-agent DQN (baseline; numpy fallback)
│       ├── fixed_curriculum.py     <- rule-based baseline
│       └── random_agent.py         <- random baseline
├── experiments/
│   └── run_experiments.py          <- runs all 5 agents × 3 profiles × 3 seeds
├── demo/
│   ├── live_demo.py                <- CLI step-by-step visualization
│   └── streamlit_app.py            <- interactive browser demo
└── results/                        <- created on first run
    ├── plots/
    │   ├── architecture.png
    │   ├── learning_curves.png
    │   ├── final_mastery_bar.png
    │   ├── policy_heatmap_*.png
    │   └── mastery_trajectory.png
    └── data/
        ├── episode_log.csv
        ├── eval_summary.csv
        └── stat_tests.csv
```

---

## Running the experiments in depth

### Quick smoke test

```bash
python experiments/run_experiments.py --quick
```

10 training + 5 eval episodes on 1 seed for each (agent, profile). Takes
~30 seconds. Verifies the pipeline works end-to-end before the full run.

### Full experiment run

```bash
python experiments/run_experiments.py \
    --train_episodes 150 \
    --eval_episodes 30 \
    --seeds 0 1 2
```

Trains all five agents (random, fixed_curriculum, LinUCB, DQN, multi_agent)
across three student profiles with three seeds each — 45 training runs, 45
evaluation runs. Takes ~3–5 minutes on CPU.

Outputs (all in `results/`, all freshly generated each run):

- `plots/learning_curves.png` — rolling-mean reward vs training episode, per profile
- `plots/final_mastery_bar.png` — eval final mastery with error bars, per agent/profile
- `plots/policy_heatmap_{linucb,dqn}_{profile}.png` — learned concept selection over the mastery grid
- `plots/mastery_trajectory.png` — mastery accumulation within an episode
- `data/episode_log.csv` — per-episode record (train + eval, every run)
- `data/eval_summary.csv` — eval means/stds per (profile, agent)
- `data/stat_tests.csv` — Welch's t-test, Cohen's d, 95% CIs vs. fixed_curriculum

---

## The demos

### Streamlit web demo

```bash
streamlit run demo/streamlit_app.py
```

Sidebar controls: student profile, training episodes, session length, per-step
pause. What you see during a run:

- Side-by-side mastery bars: the tutor's estimate (what it thinks the student
  knows) versus the student's hidden true mastery (what they actually know)
- Per-step trace showing the Coordinator's strategy, the Question Selector's
  choice, the question, the outcome, and any hint delivered
- Running accuracy plot with the ZPD band highlighted in green
- Strategy distribution over the session (how often the agent picked
  TEACH / PRACTICE / REVIEW / ASSESS)

### CLI demo

```bash
# Trained LinUCB on a beginner
python demo/live_demo.py --agent linucb --profile beginner --train_episodes 150

# Trained DQN on a gap-filled student
python demo/live_demo.py --agent dqn --profile gap_filled --train_episodes 200

# Rule-based baseline for comparison
python demo/live_demo.py --agent fixed_curriculum --profile intermediate --train_episodes 0

# Slower pace for screen recording
python demo/live_demo.py --agent linucb --profile beginner --pause 1.5
```

---

## Environment compatibility

**PyTorch is optional.** The project ships a complete pure-numpy DQN
implementation (manual backprop, Adam, Huber loss, experience replay, target
network) that activates automatically when PyTorch can't be loaded — a real
concern on Windows/Anaconda, where `c10.dll` initialization routinely fails.
No code changes required; you'll see `DQN backend: numpy` at startup
instead of `torch`.

If you want PyTorch for speed, the CPU-only wheel works for most setups:

```bash
pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
```

---

## Key references

Grounding the reward function:
- Vygotsky, L. S. (1978). *Mind in Society: The Development of Higher Psychological Processes*. Harvard University Press.
- Corbett, A. T., & Anderson, J. R. (1995). Knowledge tracing: Modeling the acquisition of procedural knowledge. *User Modeling and User-Adapted Interaction*, 4(4).

RL algorithms:
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. *WWW 2010*.
- Mnih, V. et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518.

Prior work in RL for tutoring:
- Doroudi, S., Aleven, V., & Brunskill, E. (2019). Where's the reward? A review of RL for instructional sequencing. *IJAIED*.

---

## Limitations

- **Synthetic students only.** The simulator is BKT-based, not validated
  against real learners. A small human pilot is the obvious next step.
- **Static question bank.** No LLM-based question generation; the 90-item
  bank caps the achievable diversity.
- **Short horizons.** 30-step episodes favour the bandit's immediate credit
  assignment over DQN's temporal credit assignment.
- **Goodhart risk.** Optimizing BKT mastery is a proxy for real learning;
  deployment would need retention tests and transfer tasks to catch drift.

---

## License

MIT — use it, fork it, build on it. Attribution appreciated but not required.