"""
Streamlit demo for the Adaptive SQL Tutor multi-agent system.

Run with:
    streamlit run demo/streamlit_app.py

Shows, in a browser UI:
  - Side-by-side visualization of the tutor's BKT estimate vs the student's
    true mastery (per-concept bars)
  - Each step's full agent trace: Coordinator's strategy + reason,
    Question Selector's choice + reason, question, outcome, hint (if wrong)
  - Running accuracy vs the ZPD target band
  - Auto-simulates the student using the same BKT simulator as the CLI demo
  - Trains in-session (fast — ~10s) so the user sees a warmed-up agent
"""
import os
import sys
import time

import numpy as np
import streamlit as st
import pandas as pd

# Path setup so this runs from the project root
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from src.environment import TutorEnv
from src.question_bank import CONCEPTS, N_CONCEPTS
from src.agents.session_manager import SessionManager
from src.agents.coordinator_agent import CoordinatorAgent, ACTION_NAMES
from src.agents.question_selector import QuestionSelector
from src.agents.hint_provider import HintProvider


# --------------------------------------------------------------------------- #
# Page setup                                                                  #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Adaptive SQL Tutor",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Adaptive SQL Tutor — Multi-Agent RL Demo")
st.caption(
    "Coordinator (DQN) → Question Selector (LinUCB) → Hint Provider (rule-based). "
    "Reward is BKT learning-gain + Zone-of-Proximal-Development bonus."
)

# --------------------------------------------------------------------------- #
# Sidebar controls                                                            #
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Session configuration")
    profile = st.selectbox(
        "Student profile",
        options=["beginner", "intermediate", "gap_filled"],
        index=0,
        help="beginner = low prior; intermediate = medium prior; "
             "gap_filled = strong basics, weak advanced.",
    )
    train_eps = st.slider("Training episodes (before demo)", 0, 200, 50, step=10,
                          help="More training = better policy. Auto-simulated, fast.")
    demo_steps = st.slider("Demo session length (steps)", 5, 30, 15)
    pause = st.slider("Pause between steps (seconds)", 0.0, 2.0, 0.3, step=0.1,
                      help="Slow it down for screen recording.")
    seed = st.number_input("Random seed", value=42, step=1)
    run_button = st.button("▶️ Run session", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Legend**  \n"
        "🧠 TEACH — introduce new concept  \n"
        "💪 PRACTICE — reinforce partially-learned  \n"
        "🔁 REVIEW — challenge mastered concept  \n"
        "🧪 ASSESS — probe uncertain area"
    )
    st.markdown("---")
    st.caption("Multi-agent system built on top of a BKT-based student "
               "simulator. See `report/technical_report.pdf` for details.")


# --------------------------------------------------------------------------- #
# Helper: render mastery bars                                                 #
# --------------------------------------------------------------------------- #
def render_mastery(est_mastery, true_mastery, container):
    """Side-by-side progress bars: tutor's estimate vs ground truth."""
    df = pd.DataFrame({
        "Concept": [c["name"] for c in CONCEPTS],
        "Tutor estimate": est_mastery,
        "True mastery (hidden from tutor)": true_mastery,
    })
    container.dataframe(
        df,
        column_config={
            "Tutor estimate": st.column_config.ProgressColumn(
                "Tutor estimate", format="%.2f", min_value=0.0, max_value=1.0),
            "True mastery (hidden from tutor)": st.column_config.ProgressColumn(
                "True mastery (hidden from tutor)",
                format="%.2f", min_value=0.0, max_value=1.0),
        },
        hide_index=True,
        use_container_width=True,
    )


STRATEGY_ICON = {
    "TEACH": "🧠", "PRACTICE": "💪", "REVIEW": "🔁", "ASSESS": "🧪",
}


# --------------------------------------------------------------------------- #
# Main run logic                                                              #
# --------------------------------------------------------------------------- #
if run_button:
    # Header panels
    col_left, col_right = st.columns([1, 1], gap="large")
    with col_left:
        st.subheader("📊 Student state")
        mastery_slot = st.empty()
    with col_right:
        st.subheader("🎯 Live metrics")
        metric_cols = st.columns(4)
        reward_slot = metric_cols[0].empty()
        acc_slot = metric_cols[1].empty()
        mastery_sum_slot = metric_cols[2].empty()
        step_slot = metric_cols[3].empty()
        plot_slot = st.empty()

    st.markdown("---")
    st.subheader("📝 Step-by-step trace")
    trace_slot = st.empty()

    # Build env + multi-agent session
    env = TutorEnv(profile=profile, episode_length=demo_steps, seed=int(seed))
    coord = CoordinatorAgent(state_dim=env.state_dim, seed=int(seed))
    selector = QuestionSelector(state_dim=env.state_dim, seed=int(seed))
    hinter = HintProvider()
    mgr = SessionManager(env, coord, selector, hinter, seed=int(seed))

    # Training phase
    if train_eps > 0:
        with st.spinner(f"Training Coordinator + Question Selector for "
                        f"{train_eps} episodes..."):
            env_train = TutorEnv(profile=profile, episode_length=30, seed=int(seed))
            mgr_train = SessionManager(env_train, coord, selector, hinter, seed=int(seed))
            for ep in range(train_eps):
                mgr_train.run_episode(seed=int(seed) * 1000 + ep, train=True)
        st.success(f"Trained for {train_eps} episodes. Running live demo now.")

    # Eval / demo mode
    mgr.train_mode(False)
    state = env.reset(seed=int(seed) + 9999)

    # Initial state render
    render_mastery(env.est_mastery, env.student.true_mastery, mastery_slot)
    reward_slot.metric("Total reward", "0.00")
    acc_slot.metric("Accuracy", "-")
    mastery_sum_slot.metric("Final mastery", f"{env.student.total_mastery():.2f}/10")
    step_slot.metric("Step", f"0/{demo_steps}")

    # Run live
    total_reward = 0.0
    n_correct = 0
    trace_rows = []
    reward_history = []
    accuracy_history = []

    from src.agents.session_manager import StepTrace
    from src.question_bank import action_to_concept_diff, get_question

    for step in range(demo_steps):
        # Multi-agent decision chain
        strategy, strat_reason, _ = coord.act(state, return_reason=True)
        action, sel_reason = selector.act(state, strategy, return_reason=True)

        concept, difficulty = action_to_concept_diff(action)
        question_text = get_question(concept, difficulty, idx=step)

        next_state, reward, done, info = env.step(action)
        coord.observe_outcome(info["correct"])

        hint_text, hint_type = None, None
        if not info["correct"]:
            hint_text, hint_type = hinter.get_hint(
                concept, state[concept], return_type=True)

        total_reward += reward
        n_correct += int(info["correct"])
        reward_history.append(total_reward)
        accuracy_history.append(n_correct / (step + 1))

        trace_rows.append({
            "Step": step + 1,
            "Strategy": f"{STRATEGY_ICON[ACTION_NAMES[strategy]]} {ACTION_NAMES[strategy]}",
            "Concept": CONCEPTS[concept]["name"],
            "Diff.": difficulty,
            "Question": question_text,
            "Result": "✅ correct" if info["correct"] else "❌ wrong",
            "Reward": f"{reward:+.2f}",
            "Hint (if wrong)": (hint_text or "—"),
        })

        # Live UI update
        render_mastery(env.est_mastery, env.student.true_mastery, mastery_slot)
        reward_slot.metric("Total reward", f"{total_reward:+.2f}")
        acc_slot.metric(
            "Accuracy",
            f"{n_correct / (step + 1):.0%}",
            help="ZPD optimum is ~75%. Either much higher or much lower is bad.",
        )
        mastery_sum_slot.metric(
            "True mastery",
            f"{env.student.total_mastery():.2f}/10",
        )
        step_slot.metric("Step", f"{step + 1}/{demo_steps}")

        trace_slot.dataframe(
            pd.DataFrame(trace_rows),
            hide_index=True,
            use_container_width=True,
        )

        # Running accuracy vs ZPD band plot
        if len(accuracy_history) >= 2:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 2.5))
            xs = list(range(1, len(accuracy_history) + 1))
            ax.plot(xs, accuracy_history, marker="o", label="Running accuracy")
            ax.axhspan(0.70, 0.85, color="green", alpha=0.15,
                       label="ZPD band (0.70–0.85)")
            ax.axhline(0.75, color="green", linestyle=":", alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("Step")
            ax.set_ylabel("Accuracy so far")
            ax.set_title("Is the agent keeping the student in the ZPD?")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            plot_slot.pyplot(fig, clear_figure=True)
            plt.close(fig)

        state = next_state
        if pause > 0:
            time.sleep(pause)
        if done:
            break

    # Final summary
    st.markdown("---")
    st.subheader("🏁 Session summary")
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Total reward", f"{total_reward:+.2f}")
    sc2.metric("Accuracy", f"{n_correct / demo_steps:.0%}")
    sc3.metric(
        "Final true mastery", f"{env.student.total_mastery():.2f}/10",
        delta=f"{env.student.total_mastery() - sum(c.p_init for c in env.tutor_params):+.2f} vs. start",
    )
    sc4.metric(
        "Concepts mastered (≥0.8)",
        f"{int((env.student.true_mastery >= 0.8).sum())}/10",
    )

    # Strategy distribution
    from collections import Counter
    strats = Counter([r["Strategy"].split(" ", 1)[1] for r in trace_rows])
    st.caption("**Strategy distribution this session**")
    st.bar_chart(pd.Series(strats).sort_values(ascending=False))

else:
    # Landing placeholder
    st.info(
        "👈 Set the configuration in the sidebar and click **Run session** to "
        "watch the multi-agent tutor teach a simulated student. For the video, "
        "use `Pause between steps` around 0.3–1.0s to make the live updates "
        "visible on screen."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "### How it works\n"
            "1. **Coordinator** (numpy DQN) picks a high-level strategy: "
            "TEACH / PRACTICE / REVIEW / ASSESS.\n"
            "2. **Question Selector** (LinUCB bandit) picks the specific "
            "concept and difficulty from the admissible set for that strategy.\n"
            "3. Environment simulates the student's answer using BKT; true "
            "mastery updates accordingly.\n"
            "4. **Hint Provider** (rule-based) generates a contextual hint if "
            "the student answered incorrectly.\n"
            "5. Both learning agents update from the shared ZPD-based reward."
        )
    with col_b:
        st.markdown(
            "### What you'll see\n"
            "- **Side-by-side mastery bars**: what the tutor believes vs. the "
            "hidden ground truth.\n"
            "- **Per-step trace**: every agent's decision, with the reason.\n"
            "- **Running accuracy plot**: is the agent keeping the student in "
            "the productive-struggle band?\n"
            "- **Strategy distribution**: did the agent discover a sensible "
            "pedagogical mix?"
        )
