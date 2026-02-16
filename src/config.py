"""Configuration for multi-turn regression experiments."""
import os
import random
import numpy as np

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)

# API configuration
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

# Models to test (using OpenAI models since OpenRouter budget is exceeded)
MODELS = {
    "gpt-4.1": {"provider": "openai", "model_id": "gpt-4.1"},
    "gpt-4o": {"provider": "openai", "model_id": "gpt-4o"},
    "gpt-4o-mini": {"provider": "openai", "model_id": "gpt-4o-mini"},
}

# Conversation depths to test
TURN_DEPTHS = [1, 3, 5, 10, 15, 20]

# Number of probes per condition
N_PROBES = 50

# Temperature (fixed for reproducibility)
TEMPERATURE = 0.0

# Paths
RESULTS_DIR = "results"
RAW_DIR = "results/raw"
PLOTS_DIR = "results/plots"
FIGURES_DIR = "figures"
