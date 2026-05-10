# visualize.py - All matplotlib KPI graphs

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def plot_all(kpi, ml_results, csp_normal, csp_mrv,
             search_data, victims, assignment):
    """
    Generates a single matplotlib figure with 6 KPI subplots:
      1. Victim Outcome
      2. Route Cost Comparison (BFS vs A* per victim)
      3. Search Algorithm Cost & Danger
      4. ML Model Metrics
      5. CSP Backtracking Comparison
      6. Survival Probability per Victim
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle("AIDRA — KPI Analysis Dashboard", fontsize=16, fontweight="bold", y=1.01)
    fig.patch.set_facecolor("#f4f6f9")

    colors = {
        "BFS":    "#4e79a7",
        "A*":     "#f28e2b",
        "Greedy": "#59a14f",
        "HC":     "#e15759",
    }

    victim_labels = [f"V{i+1}\n{v}" for i, v in enumerate(victims)]

    # ── 1. Victim Outcome (save rate) ────────────────────────────────────────
    ax = axes[0][0]
    ax.set_facecolor("#ffffff")
    n           = len(victims)
    saved       = kpi["victims_saved"]
    not_saved   = n - saved
    wedges, texts, autotexts = ax.pie(
        [saved, not_saved],
        labels=["Saved", "Not Reached"],
        autopct="%1.0f%%",
        colors=["#59a14f", "#e15759"],
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
    ax.set_title("KPI 1 — Victim Save Rate", fontweight="bold")

    # ── 2. Route Cost: BFS vs A* per victim ──────────────────────────────────
    ax = axes[0][1]
    ax.set_facecolor("#ffffff")
    x    = np.arange(len(victims))
    w    = 0.35
    bfs_costs = search_data["bfs_costs"]
    as_costs  = search_data["as_costs"]
    b1 = ax.bar(x - w/2, bfs_costs, w, label="BFS",  color=colors["BFS"],  edgecolor="white")
    b2 = ax.bar(x + w/2, as_costs,  w, label="A*",   color=colors["A*"],   edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(victim_labels, fontsize=8)
    ax.set_ylabel("Path Cost")
    ax.set_title("KPI 2 — Route Cost: BFS vs A* per Victim", fontweight="bold")
    ax.legend()
    ax.bar_label(b1, padding=2, fontsize=8)
    ax.bar_label(b2, padding=2, fontsize=8)

    # ── 3. Algorithm Comparison: Steps & Danger per victim (A* focus) ────────
    ax = axes[0][2]
    ax.set_facecolor("#ffffff")
    as_steps  = search_data["as_steps"]
    as_danger = search_data["as_danger"]
    x = np.arange(len(victims))
    ax.bar(x - 0.2, as_steps,  0.35, label="Steps",  color="#76b7b2", edgecolor="white")
    ax.bar(x + 0.2, as_danger, 0.35, label="Danger Cells", color="#e15759", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(victim_labels, fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title("KPI 3 — A* Steps & Danger Exposure per Victim", fontweight="bold")
    ax.legend()

    # ── 4. ML Model Metrics ───────────────────────────────────────────────────
    ax = axes[1][0]
    ax.set_facecolor("#ffffff")
    metrics    = ["Accuracy", "Precision", "Recall", "F1", "Specificity"]
    knn_vals   = [ml_results["kNN"][m]         for m in metrics]
    nb_vals    = [ml_results["Naive Bayes"][m]  for m in metrics]
    x    = np.arange(len(metrics))
    w    = 0.35
    b1 = ax.bar(x - w/2, knn_vals, w, label="kNN",         color="#4e79a7", edgecolor="white")
    b2 = ax.bar(x + w/2, nb_vals,  w, label="Naive Bayes", color="#f28e2b", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("KPI 4 — ML Model Evaluation Metrics", fontweight="bold")
    ax.legend()
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=7)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=7)

    # ── 5. CSP Backtracking Comparison ───────────────────────────────────────
    ax = axes[1][1]
    ax.set_facecolor("#ffffff")
    csp_labels   = ["Backtracks", "Total BT Calls"]
    normal_vals  = [csp_normal.backtrack_count, csp_normal.calls]
    mrv_vals     = [csp_mrv.backtrack_count,    csp_mrv.calls]
    x    = np.arange(len(csp_labels))
    w    = 0.35
    b1 = ax.bar(x - w/2, normal_vals, w, label="Normal BT", color="#9c755f", edgecolor="white")
    b2 = ax.bar(x + w/2, mrv_vals,    w, label="MRV BT",    color="#bab0ac", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(csp_labels, fontsize=10)
    ax.set_ylabel("Count")
    ax.set_title("KPI 5 — CSP: Normal vs MRV Backtracking", fontweight="bold")
    ax.legend()
    ax.bar_label(b1, padding=2, fontsize=9)
    ax.bar_label(b2, padding=2, fontsize=9)

    # ── 6. Survival Probability per Victim ───────────────────────────────────
    ax = axes[1][2]
    ax.set_facecolor("#ffffff")
    knn_surv = search_data["knn_survival"]
    nb_surv  = search_data["nb_survival"]
    x = np.arange(len(victims))
    ax.plot(x, knn_surv, "o-", color="#4e79a7", label="kNN",         linewidth=2, markersize=7)
    ax.plot(x, nb_surv,  "s-", color="#f28e2b", label="Naive Bayes", linewidth=2, markersize=7)
    ax.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Risk Threshold (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(victim_labels, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Survival Probability")
    ax.set_title("KPI 6 — Survival Probability per Victim", fontweight="bold")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("aidra_kpi_dashboard.png", dpi=150, bbox_inches="tight")
    print("\n  📊 KPI Dashboard saved → aidra_kpi_dashboard.png")
    plt.show()