# Technical Report Outline

This file is a scaffold for the PDF report required by the assignment.
Fill in each section using the plots / CSVs from `results/` after your run.

Target length: ~8–12 pages including figures.

---

## 1. Executive Summary  (~0.5 page)
One paragraph each:
- **Problem**: Adaptive tutoring systems typically optimize for accuracy,
  which incentivizes easy questions and boring sessions.
- **Approach**: RL agent selects (concept, difficulty) pairs; reward is
  BKT mastery gain + ZPD bonus − frustration/boredom penalties.
- **Result**: [Fill in with numbers from `eval_summary.csv`, e.g.
  "LinUCB outperformed the rule-based curriculum by X% final mastery
  on gap-filled students (p=0.04)."]
- **Contribution**: BKT-state + ZPD reward framing; two RL methods evaluated
  across three student profiles with statistical validation.

---

## 2. System Architecture  (~1 page, include diagram)

Architecture diagram to draw (use Excalidraw or Mermaid):

```
  ┌──────────────────────┐    action (concept, difficulty)    ┌─────────────────────┐
  │      RL Agent        │ ──────────────────────────────────▶│   TutorEnv          │
  │  (LinUCB or DQN)     │                                    │                     │
  │                      │◀── BKT mastery vector (10-d) ──────│   ┌─────────────┐   │
  │                      │    reward (learning gain + ZPD)    │   │  Student    │   │
  └──────────────────────┘                                    │   │  Simulator  │   │
                                                              │   │  (BKT-based)│   │
                                                              │   └─────────────┘   │
                                                              │   Question Bank     │
                                                              │   (90 SQL items)    │
                                                              └─────────────────────┘
```

Sub-sections to write:
- **2.1** Student simulator (reference `src/student_simulator.py`; describe the
  three profiles and the BKT parameters).
- **2.2** BKT update rule and how tutor-estimated vs true mastery diverge.
- **2.3** Environment, state space, action space (10-d state, 30 actions).
- **2.4** Reward function (reproduce the formula from `src/environment.py` —
  mastery gain × 10, ZPD bonus +0.3 when p_success ∈ [0.70, 0.85],
  frustration −0.5 after 3+ consecutive fails, boredom −0.4 on mastered
  concepts).

---

## 3. Mathematical Formulation  (~2 pages)

### 3.1 Bayesian Knowledge Tracing
State: P(L_t) = probability student has mastered concept at time t.

Observation update (Bayes):

```
P(L_t | correct)   = P(L_t)(1 - p_slip) /
                     [P(L_t)(1 - p_slip) + (1 - P(L_t)) p_guess]
P(L_t | incorrect) = P(L_t) p_slip /
                     [P(L_t) p_slip + (1 - P(L_t))(1 - p_guess)]
```

Learning transition: P(L_{t+1}) = P(L_t | obs) + (1 − P(L_t | obs)) · p_learn

### 3.2 LinUCB contextual bandit
For arm a ∈ {0..29}, context x ∈ ℝ^10:
- A_a := I + Σ x_t x_t^T
- b_a := Σ r_t x_t
- θ_a := A_a^{-1} b_a
- Action: argmax_a [ θ_a^T x + α · √(x^T A_a^{-1} x) ] with α=0.6

### 3.3 DQN
Q(s, a; θ) ≈ 2-layer MLP [10 → 128 → 128 → 30].
Loss: L(θ) = E[(r + γ max_{a'} Q(s', a'; θ⁻) − Q(s, a; θ))²]
with γ = 0.95, target-net sync every 200 steps, Huber loss,
replay buffer 20k, batch 64, ε-greedy decayed over 6000 steps.

### 3.4 Reward engineering
r_t = 10 · Δ(Σ P(L)) + 0.3·𝟙[p_success ∈ [0.70, 0.85]] − 0.5·𝟙[consec_fail ≥ 3]
     − 0.4·𝟙[concept mastered AND global avg mastery > 0.5 AND diff ≠ hard]

Cite Vygotsky (ZPD) and Corbett & Anderson (BKT) here.

---

## 4. Experimental Methodology  (~1 page)

- **Agents**: Random, FixedCurriculum (topological order + difficulty ladder),
  LinUCB, DQN.
- **Student profiles**: beginner (low prior), intermediate (mid prior),
  gap_filled (strong basics, weak advanced).
- **Training**: 150 episodes per (agent, profile, seed).
- **Evaluation**: 30 episodes per (agent, profile, seed) with exploration off.
- **Seeds**: [0, 1, 2].
- **Metrics**: episode reward, final true mastery (ground truth unseen by
  agent), accuracy (proxy), time-to-mastery (first step with Σ est mastery ≥ 4).
- **Statistical tests**: Welch's t-test for each RL agent vs FixedCurriculum
  on final true mastery (results in `results/data/stat_tests.csv`).

---

## 5. Results  (~2 pages; embed the four plots)

### 5.1 Learning curves (`results/plots/learning_curves.png`)
[Describe the trajectory: LinUCB converges fast, DQN is slower but reaches
higher asymptote, baselines are flat.]

### 5.2 Final mastery (`results/plots/final_mastery_bar.png`)
Report exact means ± std from `eval_summary.csv`.

### 5.3 Policy visualization (`results/plots/policy_heatmap_*.png`)
Show which concept the agent chooses across the mastery grid. Interpret: e.g.
"At low basics mastery, LinUCB correctly picks SELECT/WHERE; only once basics
mastery > 0.6 does it move to JOINS/SUBQUERIES — the agent has discovered the
dependency structure without being told."

### 5.4 Within-episode dynamics (`results/plots/mastery_trajectory.png`)
[Describe: learning agents build mastery faster per step than random.]

### 5.5 Statistical significance (`results/data/stat_tests.csv`)
Report the t-statistic, p-value, and mean difference for each comparison.
Flag significant effects (p < 0.05).

---

## 6. Discussion  (~1 page)

### 6.1 What worked
- BKT state gave agents a compact, informative observation.
- The ZPD-based reward kept sessions in the productive-struggle band
  (report mean accuracy — should be near 0.75 for well-tuned agents).

### 6.2 What didn't, and why
- [If LinUCB underperforms fixed_curriculum: hand-coded dependency graph is
  strong prior knowledge; the bandit has to discover it from scratch.]
- [If DQN is unstable: 30 discrete actions × 10-d state is still narrow; more
  seeds would reduce variance.]

### 6.3 Connection to theory
- ZPD (Vygotsky) as a reward signal is an explicit operationalization of
  pedagogical theory rather than a proxy.
- BKT is a well-established latent-state model; using it as agent state is a
  partial-observability solution that doesn't require POMDP machinery.

---

## 7. Challenges & Solutions  (~0.5 page)
- **Reward shaping**: initial pure-accuracy reward led to agents picking only
  easy questions on already-mastered concepts. Replaced with mastery-gain +
  ZPD bonus + penalties.
- **State representation**: last-N-correct loses information; switched to BKT
  mastery vector.
- **Baseline strength**: rule-based curriculum is very strong because it
  encodes dependencies. This is a realistic bar — beating it matters.

---

## 8. Future Work  (~0.5 page)
- **Real-user pilot** (n=10–30) to validate the simulator.
- **LLM-generated questions** with verification — removes static bank limit.
- **Deep Knowledge Tracing (DKT)** LSTM to replace BKT for richer state.
- **Spaced repetition** between sessions (forgetting curve).
- **Pedagogical strategy selection** as a second-level action: not just
  (concept, difficulty) but also (worked-example vs hint vs retrieval).

---

## 9. Ethical Considerations  (~0.5 page)
- **Optimization targets**: optimizing for metric (mastery) rather than real
  learning risks Goodhart's law — students may pass BKT's model while
  lacking transfer. Mitigation: human evaluation, spaced retention tests.
- **Fairness**: results should be reported per profile, not just averaged,
  to surface any profile the agent fails on.
- **Autonomy and dependence**: adaptive tutors can reduce a learner's own
  metacognition ("which topic should I study next?") by making those
  decisions for them. Long-term use should be paired with learner-controlled
  modes.
- **Data privacy**: in a real deployment, BKT states are sensitive learning
  records; must be handled per FERPA/GDPR.

---

## 10. References
(BibTeX-style list — use the citations from the README, plus any others you add.)
