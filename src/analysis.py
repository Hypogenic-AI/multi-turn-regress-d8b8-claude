"""Analysis and visualization for multi-turn regression experiments."""
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
from scipy import stats
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import RESULTS_DIR, PLOTS_DIR, FIGURES_DIR


def load_results(filepath: str = None) -> pd.DataFrame:
    """Load experiment results into a DataFrame."""
    if filepath is None:
        filepath = f"{RESULTS_DIR}/raw/all_results.json"
    with open(filepath) as f:
        data = json.load(f)
    return pd.DataFrame(data)


def compute_alignment_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute alignment scores per model/turn_depth/probe_type."""
    results = []

    # Instruction following & constraint adherence: use 'passed' column
    for probe_type in ["instruction_following", "constraint_adherence"]:
        subset = df[df["probe_type"] == probe_type].copy()
        if subset.empty:
            continue
        grouped = subset.groupby(["model", "turn_depth"]).agg(
            pass_rate=("passed", "mean"),
            n=("passed", "count"),
        ).reset_index()
        grouped["probe_type"] = probe_type
        results.append(grouped)

    # Sycophancy: use flip rate and final accuracy
    syc = df[df["probe_type"] == "sycophancy"].copy()
    if not syc.empty:
        syc_grouped = syc.groupby(["model", "turn_depth"]).agg(
            flip_rate=("flipped", "mean"),
            initial_accuracy=("initial_correct", "mean"),
            final_accuracy=("final_correct", "mean"),
            n=("flipped", "count"),
        ).reset_index()
        syc_grouped["probe_type"] = "sycophancy"
        results.append(syc_grouped)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


def plot_alignment_degradation(df: pd.DataFrame, output_dir: str = None):
    """Plot alignment score degradation curves across turns."""
    if output_dir is None:
        output_dir = PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    scores = compute_alignment_scores(df)
    if scores.empty:
        print("No scores to plot")
        return

    # Plot 1: Instruction Following Rate by Turn Depth
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, probe_type in enumerate(["instruction_following", "constraint_adherence", "sycophancy"]):
        ax = axes[i]
        subset = scores[scores["probe_type"] == probe_type]

        if subset.empty:
            ax.set_title(f"{probe_type}\n(no data)")
            continue

        if probe_type != "sycophancy":
            for model in subset["model"].unique():
                model_data = subset[subset["model"] == model].sort_values("turn_depth")
                ax.plot(
                    model_data["turn_depth"],
                    model_data["pass_rate"],
                    marker="o",
                    label=model,
                    linewidth=2,
                )
            ax.set_ylabel("Pass Rate")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title(f"{probe_type.replace('_', ' ').title()}\nAlignment Score by Turn Depth")
        else:
            for model in subset["model"].unique():
                model_data = subset[subset["model"] == model].sort_values("turn_depth")
                ax.plot(
                    model_data["turn_depth"],
                    model_data["flip_rate"],
                    marker="s",
                    label=f"{model} (flip)",
                    linewidth=2,
                )
            ax.set_ylabel("Flip Rate (Sycophancy)")
            ax.set_ylim(-0.05, 1.05)
            ax.set_title("Sycophancy (BoolQ Challenge)\nFlip Rate by Turn Depth")

        ax.set_xlabel("Turn Depth")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/alignment_degradation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/alignment_degradation.png")

    # Plot 2: Combined alignment score (average of instruction + constraint)
    fig, ax = plt.subplots(figsize=(10, 6))

    combined = scores[scores["probe_type"].isin(["instruction_following", "constraint_adherence"])]
    if not combined.empty:
        combined_avg = combined.groupby(["model", "turn_depth"]).agg(
            alignment_score=("pass_rate", "mean"),
        ).reset_index()

        for model in combined_avg["model"].unique():
            model_data = combined_avg[combined_avg["model"] == model].sort_values("turn_depth")
            ax.plot(
                model_data["turn_depth"],
                model_data["alignment_score"],
                marker="o",
                label=model,
                linewidth=2.5,
                markersize=8,
            )

    ax.set_xlabel("Conversation Turn Depth", fontsize=12)
    ax.set_ylabel("Combined Alignment Score", fontsize=12)
    ax.set_title("Alignment Score Degradation Over Multi-Turn Conversations", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/combined_alignment_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/combined_alignment_score.png")


def plot_sycophancy_detail(df: pd.DataFrame, output_dir: str = None):
    """Detailed sycophancy analysis plots."""
    if output_dir is None:
        output_dir = PLOTS_DIR
    os.makedirs(output_dir, exist_ok=True)

    syc = df[df["probe_type"] == "sycophancy"].copy()
    if syc.empty:
        print("No sycophancy data to plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Flip rate by turn depth
    ax = axes[0]
    for model in syc["model"].unique():
        model_data = syc[syc["model"] == model].groupby("turn_depth").agg(
            flip_rate=("flipped", "mean"),
            se=("flipped", lambda x: x.std() / np.sqrt(len(x)) if len(x) > 1 else 0),
        ).reset_index()

        ax.errorbar(
            model_data["turn_depth"],
            model_data["flip_rate"],
            yerr=model_data["se"] * 1.96,
            marker="o",
            label=model,
            linewidth=2,
            capsize=3,
        )

    ax.set_xlabel("Turn Depth")
    ax.set_ylabel("Flip Rate")
    ax.set_title("Sycophancy: Flip Rate by Turn Depth\n(Higher = More Sycophantic)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Initial vs final accuracy
    ax = axes[1]
    scores = compute_alignment_scores(df)
    syc_scores = scores[scores["probe_type"] == "sycophancy"]
    if not syc_scores.empty:
        for model in syc_scores["model"].unique():
            model_data = syc_scores[syc_scores["model"] == model].sort_values("turn_depth")
            ax.plot(
                model_data["turn_depth"],
                model_data["initial_accuracy"],
                marker="o",
                linestyle="--",
                label=f"{model} (initial)",
                linewidth=1.5,
            )
            ax.plot(
                model_data["turn_depth"],
                model_data["final_accuracy"],
                marker="s",
                label=f"{model} (after challenge)",
                linewidth=2,
            )

    ax.set_xlabel("Turn Depth")
    ax.set_ylabel("Accuracy")
    ax.set_title("Factual Accuracy: Before and After Challenge")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/sycophancy_detail.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/sycophancy_detail.png")


def run_statistical_tests(df: pd.DataFrame) -> dict:
    """Run statistical tests on experiment results."""
    results = {}

    # Test 1: Does pass rate decrease with turn depth? (Spearman correlation)
    for probe_type in ["instruction_following", "constraint_adherence"]:
        subset = df[df["probe_type"] == probe_type]
        if subset.empty:
            continue

        for model in subset["model"].unique():
            model_data = subset[subset["model"] == model]
            if len(model_data) < 5:
                continue

            # Convert passed to numeric
            passed = model_data["passed"].astype(float)
            depths = model_data["turn_depth"].astype(float)

            rho, p_val = stats.spearmanr(depths, passed)
            results[f"{probe_type}_{model}_spearman"] = {
                "rho": float(rho),
                "p_value": float(p_val),
                "n": int(len(model_data)),
                "interpretation": "significant decline" if (p_val < 0.05 and rho < 0) else
                                 "significant increase" if (p_val < 0.05 and rho > 0) else
                                 "no significant trend",
            }

    # Test 2: Does flip rate increase with turn depth?
    syc = df[df["probe_type"] == "sycophancy"]
    if not syc.empty:
        for model in syc["model"].unique():
            model_data = syc[syc["model"] == model]
            if len(model_data) < 5:
                continue

            flipped = model_data["flipped"].astype(float)
            depths = model_data["turn_depth"].astype(float)

            rho, p_val = stats.spearmanr(depths, flipped)
            results[f"sycophancy_{model}_spearman"] = {
                "rho": float(rho),
                "p_value": float(p_val),
                "n": int(len(model_data)),
                "interpretation": "significant increase" if (p_val < 0.05 and rho > 0) else
                                 "significant decrease" if (p_val < 0.05 and rho < 0) else
                                 "no significant trend",
            }

    # Test 3: Early vs. Late turn comparison (Mann-Whitney U)
    for probe_type in ["instruction_following", "constraint_adherence"]:
        subset = df[df["probe_type"] == probe_type]
        if subset.empty:
            continue

        for model in subset["model"].unique():
            model_data = subset[subset["model"] == model]
            early = model_data[model_data["turn_depth"] <= 3]["passed"].astype(float)
            late = model_data[model_data["turn_depth"] >= 10]["passed"].astype(float)

            if len(early) < 3 or len(late) < 3:
                continue

            U, p_val = stats.mannwhitneyu(early, late, alternative="greater")
            effect_size = U / (len(early) * len(late))  # rank-biserial correlation

            results[f"{probe_type}_{model}_early_vs_late"] = {
                "early_mean": float(early.mean()),
                "late_mean": float(late.mean()),
                "U_statistic": float(U),
                "p_value": float(p_val),
                "effect_size": float(effect_size),
                "n_early": int(len(early)),
                "n_late": int(len(late)),
                "interpretation": "early significantly better" if p_val < 0.05 else "no significant difference",
            }

    return results


def compute_degradation_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Compute linear degradation rates per model and probe type."""
    results = []

    for probe_type in ["instruction_following", "constraint_adherence"]:
        subset = df[df["probe_type"] == probe_type]
        if subset.empty:
            continue

        for model in subset["model"].unique():
            model_data = subset[subset["model"] == model]
            grouped = model_data.groupby("turn_depth")["passed"].mean().reset_index()

            if len(grouped) < 3:
                continue

            slope, intercept, r_value, p_value, std_err = stats.linregress(
                grouped["turn_depth"].values.astype(float),
                grouped["passed"].values.astype(float),
            )

            results.append({
                "model": model,
                "probe_type": probe_type,
                "slope": slope,
                "intercept": intercept,
                "r_squared": r_value ** 2,
                "p_value": p_value,
                "std_err": std_err,
            })

    return pd.DataFrame(results)


def plot_degradation_rates(df: pd.DataFrame, output_dir: str = None):
    """Plot degradation rate comparison across models."""
    if output_dir is None:
        output_dir = PLOTS_DIR

    rates = compute_degradation_rates(df)
    if rates.empty:
        print("No degradation rates to plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Combine instruction following and constraint adherence
    for probe_type in rates["probe_type"].unique():
        type_data = rates[rates["probe_type"] == probe_type]
        x = range(len(type_data))
        bars = ax.bar(
            [i + (0.35 if probe_type == "constraint_adherence" else 0) for i in x],
            type_data["slope"] * 10,  # slope per 10 turns
            width=0.35,
            label=probe_type.replace("_", " ").title(),
            alpha=0.8,
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Alignment Score Change per 10 Turns")
    ax.set_title("Alignment Degradation Rate by Model\n(More negative = faster degradation)")
    ax.set_xticks(range(len(rates["model"].unique())))
    ax.set_xticklabels(rates["model"].unique(), rotation=15, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/degradation_rates.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir}/degradation_rates.png")


def plot_heatmap(df: pd.DataFrame, output_dir: str = None):
    """Create a heatmap of pass rates by model, turn depth, and probe type."""
    if output_dir is None:
        output_dir = PLOTS_DIR

    for probe_type in ["instruction_following", "constraint_adherence"]:
        subset = df[df["probe_type"] == probe_type]
        if subset.empty:
            continue

        pivot = subset.groupby(["model", "turn_depth"])["passed"].mean().unstack(fill_value=0).astype(float)

        fig, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=0,
            vmax=1,
            ax=ax,
            cbar_kws={"label": "Pass Rate"},
        )
        ax.set_title(f"{probe_type.replace('_', ' ').title()}: Pass Rate Heatmap")
        ax.set_xlabel("Turn Depth")
        ax.set_ylabel("Model")

        plt.tight_layout()
        plt.savefig(f"{output_dir}/heatmap_{probe_type}.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {output_dir}/heatmap_{probe_type}.png")


def generate_summary_table(df: pd.DataFrame) -> str:
    """Generate a markdown summary table of results."""
    scores = compute_alignment_scores(df)
    if scores.empty:
        return "No results to summarize."

    lines = []
    lines.append("## Summary Results\n")

    # Instruction following
    for probe_type in ["instruction_following", "constraint_adherence"]:
        subset = scores[scores["probe_type"] == probe_type]
        if subset.empty:
            continue

        lines.append(f"\n### {probe_type.replace('_', ' ').title()}\n")
        lines.append("| Model | Turn 1 | Turn 3 | Turn 5 | Turn 10 | Turn 15 | Turn 20 |")
        lines.append("|-------|--------|--------|--------|---------|---------|---------|")

        for model in subset["model"].unique():
            model_data = subset[subset["model"] == model]
            row = f"| {model} |"
            for depth in [1, 3, 5, 10, 15, 20]:
                val = model_data[model_data["turn_depth"] == depth]["pass_rate"].values
                if len(val) > 0:
                    row += f" {val[0]:.2f} |"
                else:
                    row += " - |"
            lines.append(row)

    # Sycophancy
    syc = scores[scores["probe_type"] == "sycophancy"]
    if not syc.empty:
        lines.append(f"\n### Sycophancy (Flip Rate)\n")
        lines.append("| Model | Turn 1 | Turn 3 | Turn 5 | Turn 10 | Turn 15 | Turn 20 |")
        lines.append("|-------|--------|--------|--------|---------|---------|---------|")

        for model in syc["model"].unique():
            model_data = syc[syc["model"] == model]
            row = f"| {model} |"
            for depth in [1, 3, 5, 10, 15, 20]:
                val = model_data[model_data["turn_depth"] == depth]["flip_rate"].values
                if len(val) > 0:
                    row += f" {val[0]:.2f} |"
                else:
                    row += " - |"
            lines.append(row)

    return "\n".join(lines)


def full_analysis(results_path: str = None):
    """Run complete analysis pipeline."""
    df = load_results(results_path)
    print(f"Loaded {len(df)} results")
    print(f"Models: {df['model'].unique()}")
    print(f"Probe types: {df['probe_type'].unique()}")
    print(f"Turn depths: {sorted(df['turn_depth'].unique())}")

    # Generate plots
    print("\n--- Generating plots ---")
    plot_alignment_degradation(df)
    plot_sycophancy_detail(df)
    plot_degradation_rates(df)
    plot_heatmap(df)

    # Statistical tests
    print("\n--- Running statistical tests ---")
    stat_results = run_statistical_tests(df)
    with open(f"{RESULTS_DIR}/statistical_tests.json", "w") as f:
        json.dump(stat_results, f, indent=2)
    print(f"Statistical test results saved to {RESULTS_DIR}/statistical_tests.json")

    for key, val in stat_results.items():
        rho = val.get('rho', 'N/A')
        p = val.get('p_value', 'N/A')
        rho_str = f"{rho:.3f}" if isinstance(rho, (int, float)) and not np.isnan(rho) else str(rho)
        p_str = f"{p:.4f}" if isinstance(p, (int, float)) and not np.isnan(p) else str(p)
        interp = val.get('interpretation', '')
        print(f"  {key}: rho={rho_str}, p={p_str} -> {interp}")

    # Degradation rates
    print("\n--- Computing degradation rates ---")
    rates = compute_degradation_rates(df)
    if not rates.empty:
        print(rates.to_string())
        rates.to_csv(f"{RESULTS_DIR}/degradation_rates.csv", index=False)

    # Summary table
    print("\n--- Summary Table ---")
    summary = generate_summary_table(df)
    print(summary)
    with open(f"{RESULTS_DIR}/summary_table.md", "w") as f:
        f.write(summary)

    return df, stat_results, rates


if __name__ == "__main__":
    full_analysis()
