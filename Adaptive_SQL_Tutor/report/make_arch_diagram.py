"""
Render the system architecture diagram as a PNG. Called by generate_report.py
but can also be run standalone:  python report/make_arch_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS_DIR = os.path.join(ROOT, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def render(out_path=None):
    if out_path is None:
        out_path = os.path.join(PLOTS_DIR, "architecture.png")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.axis("off")

    def box(x, y, w, h, label, color, txt_kwargs=None):
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
            linewidth=1.6, edgecolor="#2a2a2a", facecolor=color,
        )
        ax.add_patch(rect)
        tk = dict(ha="center", va="center", fontsize=10, fontweight="bold")
        if txt_kwargs:
            tk.update(txt_kwargs)
        ax.text(x + w / 2, y + h / 2, label, **tk)

    # --- Session Manager container (big outer box for the multi-agent side) ---
    outer = patches.FancyBboxPatch(
        (0.2, 3.4), 6.3, 4.9,
        boxstyle="round,pad=0.15,rounding_size=0.2",
        linewidth=2.0, edgecolor="#1f4e8a", facecolor="#eaf3ff",
    )
    ax.add_patch(outer)
    ax.text(3.35, 8.0, "SESSION MANAGER  (multi-agent orchestrator)",
            ha="center", fontsize=11.5, fontweight="bold", color="#1f4e8a")

    # --- 4 specialized agents inside the Session Manager ---
    box(0.5, 6.0, 2.85, 1.6,
        "COORDINATOR\n(DQN)\n\nTEACH / PRACTICE\nREVIEW / ASSESS",
        "#c9e2ff", dict(fontsize=9))
    box(3.55, 6.0, 2.75, 1.6,
        "KNOWLEDGE TRACKER\n(BKT)\n\nP(L_c | observations)\nfor each concept",
        "#d6f5d6", dict(fontsize=9))
    box(0.5, 3.8, 2.85, 1.8,
        "QUESTION SELECTOR\n(LinUCB bandit)\n\nconstrained by\ncoordinator strategy\n(concept, difficulty)",
        "#fff4c9", dict(fontsize=9))
    box(3.55, 3.8, 2.75, 1.8,
        "HINT PROVIDER\n(rule-based)\n\n3 scaffolding levels:\nreview / procedural\n/ nudge",
        "#fde2e2", dict(fontsize=9))

    # inner arrows (strategy flow)
    ax.annotate("", xy=(1.92, 5.6), xytext=(1.92, 6.0),
                arrowprops=dict(arrowstyle="->", color="#1f4e8a", lw=1.4))
    ax.text(2.2, 5.78, "strategy", ha="left", fontsize=8, color="#1f4e8a")

    # --- TutorEnv (right) ---
    box(7.4, 4.8, 4.3, 2.0,
        "TutorEnv\n\nstate = BKT mastery (10-d)\nreward = learning gain + ZPD\n                − frustration − boredom",
        "#ffe1b3", dict(fontsize=9.5, fontweight="normal"))

    # --- Student Simulator & Question Bank (bottom right) ---
    box(7.4, 1.4, 2.0, 2.4,
        "Student\nSimulator\n(BKT-based)\n\n3 profiles",
        "#e5d9f2", dict(fontsize=9.5))
    box(9.7, 1.4, 2.0, 2.4,
        "Question\nBank\n\n10 concepts\n×\n3 difficulties\n×\n3 questions",
        "#f9e4d4", dict(fontsize=9.5))

    # Env <-> Student/Bank
    ax.annotate("", xy=(8.4, 3.8), xytext=(8.4, 4.8),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))
    ax.annotate("", xy=(10.7, 3.8), xytext=(10.7, 4.8),
                arrowprops=dict(arrowstyle="<->", color="#444", lw=1.3))

    # --- Session Manager -> TutorEnv (action) ---
    ax.annotate("", xy=(7.4, 6.0), xytext=(6.5, 6.0),
                arrowprops=dict(arrowstyle="->", color="#1f4e8a", lw=2.2))
    ax.text(6.95, 6.3, "action", ha="center", fontsize=10, color="#1f4e8a",
            fontweight="bold")

    # --- TutorEnv -> Session Manager (state + reward) ---
    ax.annotate("", xy=(6.5, 5.3), xytext=(7.4, 5.3),
                arrowprops=dict(arrowstyle="->", color="#8a1f2a", lw=2.2))
    ax.text(6.95, 5.0, "state+reward", ha="center", fontsize=10,
            color="#8a1f2a", fontweight="bold")

    # Title + caption
    ax.text(6.0, 8.6, "Adaptive SQL Tutor — Multi-Agent RL Architecture",
            ha="center", fontsize=13, fontweight="bold")
    ax.text(6.0, 0.65,
            "Four agents share a BKT state; Coordinator (DQN) picks "
            "pedagogical strategy, Question Selector (LinUCB) picks the "
            "specific question given the strategy, Hint Provider scaffolds "
            "on wrong answers. Orchestrated by the Session Manager.",
            ha="center", fontsize=9, style="italic", color="#555",
            wrap=True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    render()
