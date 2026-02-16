# Do Multi-Turn Conversations Regress to the Prior?

An empirical investigation of whether LLM alignment degrades over multi-turn conversations, using behavioral probing of frontier models.

## Key Findings

- **Alignment is stable across 20 turns**: Instruction following, constraint adherence, and system instruction persistence show no statistically significant degradation (Spearman rho: -0.11 to 0.0, all p > 0.38) for GPT-4.1, GPT-4o, and GPT-4o-mini
- **Sycophancy is the primary vulnerability**: All models flip correct answers 67% of the time under progressive adversarial challenge, but this is **not turn-depth dependent**
- **Sycophancy is alignment-specific, not regression to base**: Alignment reminders reduce sycophancy by 68%, while context summaries paradoxically *increase* it by 63%
- **Stronger models are more sycophantic**: GPT-4.1 flips at weaker challenge levels (avg 0.7) than GPT-4o (avg 1.7), consistent with alignment training creating user-deference bias
- **The "regression to prior" hypothesis is not supported** for current frontier models with conversations up to 20 turns

## Methodology

Three experiments with 687 total API conversations (~3000+ API calls):

1. **Basic Alignment Battery** (450 runs): 25 behavioral probes x 6 turn depths x 3 models
2. **Hard Probes** (141 runs): Instruction persistence, progressive persuasion, boundary push
3. **Intervention Diagnostics** (96 runs): Alignment reminders vs. context summaries at turn 15

## How to Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai numpy pandas matplotlib scipy seaborn datasets tqdm

# Set API keys
export OPENAI_API_KEY="your-key"

# Run experiments
python src/run_experiments.py --n-probes 10        # Experiment 1
python src/hard_probes_experiment.py                # Experiment 2
python src/intervention_experiment.py               # Experiment 3

# Run analysis
python src/analysis.py
python src/analysis_hard_probes.py
```

## File Structure

```
├── REPORT.md           # Full research report with results
├── planning.md         # Research plan and hypothesis decomposition
├── src/                # All experiment and analysis code
│   ├── run_experiments.py       # Main experiment
│   ├── hard_probes_experiment.py # Hard probes
│   ├── intervention_experiment.py # Intervention diagnostics
│   ├── analysis.py              # Analysis for Exp 1
│   └── analysis_hard_probes.py  # Analysis for Exps 2-3
├── results/            # All results, stats, and plots
├── literature_review.md # Literature review (20+ papers)
└── resources.md        # Resource catalog
```

See [REPORT.md](REPORT.md) for full details.
