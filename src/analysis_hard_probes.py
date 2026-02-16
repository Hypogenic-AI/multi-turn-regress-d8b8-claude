"""Analysis and visualization for hard probe experiments."""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, PLOTS_DIR


def load_hard_probes(filepath=None):
    if filepath is None:
        filepath = f"{RESULTS_DIR}/raw/hard_probes_results.json"
    with open(filepath) as f:
        return json.load(f)


def load_intervention(filepath=None):
    if filepath is None:
        filepath = f"{RESULTS_DIR}/raw/intervention_results.json"
    with open(filepath) as f:
        return json.load(f)


def analyze_instruction_persistence(data):
    """Analyze and plot instruction persistence results."""
    results = data["instruction_persistence"]
    df = pd.DataFrame(results)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot: pass rate by turn depth for each model/instruction combo
    models = df["model"].unique()
    instructions = df["instruction_name"].unique()

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]
        mdf = df[df["model"] == model]

        for instr in instructions:
            idf = mdf[mdf["instruction_name"] == instr]
            idf_sorted = idf.sort_values("turn_depth")
            passed_rate = idf_sorted.groupby("turn_depth")["final_passed"].mean()

            ax.plot(
                passed_rate.index,
                passed_rate.values,
                marker="o",
                label=instr,
                linewidth=2,
            )

        ax.set_title(f"{model}", fontsize=12)
        ax.set_xlabel("Turn Depth")
        if idx == 0:
            ax.set_ylabel("Instruction Compliance Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.suptitle("System Instruction Persistence Across Conversation Turns", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/instruction_persistence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/instruction_persistence.png")

    # Aggregate across instructions
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in models:
        mdf = df[df["model"] == model]
        avg_by_depth = mdf.groupby("turn_depth")["final_passed"].agg(["mean", "sem"]).reset_index()

        ax.errorbar(
            avg_by_depth["turn_depth"],
            avg_by_depth["mean"],
            yerr=avg_by_depth["sem"] * 1.96,
            marker="o",
            label=model,
            linewidth=2.5,
            capsize=4,
            markersize=8,
        )

    ax.set_xlabel("Conversation Turn Depth", fontsize=12)
    ax.set_ylabel("Average Instruction Compliance Rate", fontsize=12)
    ax.set_title("System Instruction Persistence\n(Averaged Across 5 Instruction Types)", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/instruction_persistence_aggregate.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/instruction_persistence_aggregate.png")

    # Statistical tests
    stat_results = {}
    for model in models:
        mdf = df[df["model"] == model]
        passed = mdf["final_passed"].astype(float)
        depths = mdf["turn_depth"].astype(float)
        if len(mdf) >= 5:
            rho, p = stats.spearmanr(depths, passed)
            stat_results[f"{model}_spearman"] = {
                "rho": float(rho), "p_value": float(p),
                "interpretation": "significant decline" if (p < 0.05 and rho < 0)
                    else "no significant trend"
            }

            # Early vs late comparison
            early = mdf[mdf["turn_depth"] <= 3]["final_passed"].astype(float)
            late = mdf[mdf["turn_depth"] >= 15]["final_passed"].astype(float)
            if len(early) >= 3 and len(late) >= 3:
                U, p_u = stats.mannwhitneyu(early, late, alternative="greater")
                stat_results[f"{model}_early_vs_late"] = {
                    "early_mean": float(early.mean()),
                    "late_mean": float(late.mean()),
                    "U": float(U), "p_value": float(p_u),
                    "interpretation": "early significantly better" if p_u < 0.05 else "no significant difference"
                }

    return df, stat_results


def analyze_progressive_persuasion(data):
    """Analyze progressive persuasion results."""
    results = data["progressive_persuasion"]
    df = pd.DataFrame(results)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Plot: flip rate and average flip level by model
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    models = df["model"].unique()

    # Flip rate
    ax = axes[0]
    flip_rates = []
    for model in models:
        mdf = df[df["model"] == model]
        initially_correct = mdf[mdf["initial_correct"] == True]
        flip_rate = (initially_correct["flip_level"].notna().sum() / len(initially_correct)) if len(initially_correct) > 0 else 0
        flip_rates.append(flip_rate)

    bars = ax.bar(models, flip_rates, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set_ylabel("Flip Rate Under Progressive Challenge")
    ax.set_title("Sycophancy Under Progressive Persuasion")
    ax.set_ylim(0, 1)
    for bar, rate in zip(bars, flip_rates):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Average flip level (among those that flipped)
    ax = axes[1]
    avg_levels = []
    for model in models:
        mdf = df[df["model"] == model]
        flipped = mdf[mdf["flip_level"].notna()]
        avg = flipped["flip_level"].mean() if len(flipped) > 0 else 0
        avg_levels.append(avg)

    bars = ax.bar(models, avg_levels, color=["#2196F3", "#FF9800", "#4CAF50"])
    ax.set_ylabel("Average Challenge Level at Flip\n(0=weakest, 4=strongest)")
    ax.set_title("Resistance Level Before Flip\n(Higher = More Resistant)")
    ax.set_ylim(0, 5)
    for bar, level in zip(bars, avg_levels):
        if level > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{level:.1f}', ha='center', va='bottom', fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/progressive_persuasion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/progressive_persuasion.png")

    return df


def analyze_intervention(data):
    """Analyze intervention experiment results."""
    df = pd.DataFrame(data)

    os.makedirs(PLOTS_DIR, exist_ok=True)

    # Instruction following intervention effects
    if_data = df[df["probe_type"] == "instruction_following"]
    syc_data = df[df["probe_type"] == "sycophancy"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Instruction following
    ax = axes[0]
    if not if_data.empty:
        for model in if_data["model"].unique():
            mdf = if_data[if_data["model"] == model]
            rates = mdf.groupby("condition")["passed"].mean()
            conditions = ["control", "alignment_reminder", "context_summary"]
            vals = [rates.get(c, 0) for c in conditions]
            x = np.arange(len(conditions))
            width = 0.25
            model_idx = list(if_data["model"].unique()).index(model)
            ax.bar(x + model_idx * width, vals, width, label=model, alpha=0.8)

        ax.set_xticks(np.arange(len(conditions)) + width)
        ax.set_xticklabels(["Control", "Alignment\nReminder", "Context\nSummary"])
        ax.set_ylabel("Pass Rate")
        ax.set_title("Instruction Following\nEffect of Interventions at Turn 15")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    # Sycophancy
    ax = axes[1]
    if not syc_data.empty:
        for model in syc_data["model"].unique():
            mdf = syc_data[syc_data["model"] == model]
            rates = mdf.groupby("condition")["flipped"].mean()
            conditions = ["control", "alignment_reminder", "context_summary"]
            vals = [rates.get(c, 0) for c in conditions]
            x = np.arange(len(conditions))
            width = 0.25
            model_idx = list(syc_data["model"].unique()).index(model)
            ax.bar(x + model_idx * width, vals, width, label=model, alpha=0.8)

        ax.set_xticks(np.arange(len(conditions)) + width)
        ax.set_xticklabels(["Control", "Alignment\nReminder", "Context\nSummary"])
        ax.set_ylabel("Flip Rate (Sycophancy)")
        ax.set_title("Sycophancy Resistance\nEffect of Interventions at Turn 15")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/intervention_effects.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {PLOTS_DIR}/intervention_effects.png")

    return df


def run_all_analysis():
    """Run all analysis."""
    print("=" * 60)
    print("HARD PROBES ANALYSIS")
    print("=" * 60)

    # Hard probes
    try:
        hard_data = load_hard_probes()
        print("\n--- Instruction Persistence ---")
        ip_df, ip_stats = analyze_instruction_persistence(hard_data)
        for key, val in ip_stats.items():
            print(f"  {key}: {val}")

        print("\n--- Progressive Persuasion ---")
        pp_df = analyze_progressive_persuasion(hard_data)

        # Save stats
        with open(f"{RESULTS_DIR}/hard_probes_stats.json", "w") as f:
            json.dump(ip_stats, f, indent=2)
    except FileNotFoundError:
        print("Hard probes results not found yet.")

    # Intervention
    try:
        interv_data = load_intervention()
        print("\n--- Intervention Effects ---")
        interv_df = analyze_intervention(interv_data)
    except FileNotFoundError:
        print("Intervention results not found yet.")


if __name__ == "__main__":
    run_all_analysis()
