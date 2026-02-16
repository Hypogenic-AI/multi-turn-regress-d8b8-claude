# Datasets for Multi-Turn Conversation Regression Research

This directory contains datasets used in the research project: "Do Multi-Turn Conversations Regress to the Prior?"

## Setup

```bash
# Activate the virtual environment
source .venv/bin/activate

# Install the datasets library
uv pip install datasets
```

## Tier 1 Datasets

### 1. Multi-IF (Facebook)

- **Source**: [facebook/Multi-IF](https://huggingface.co/datasets/facebook/Multi-IF)
- **Status**: Downloaded (full dataset)
- **Size**: ~2 MB on disk
- **Split**: train (4,501 examples)
- **Features**: `turns`, `responses`, `turn_1_prompt`, `turn_1_instruction_id_list`, `turn_1_kwargs`, `turn_2_prompt`, `turn_2_instruction_id_list`, `turn_2_kwargs`, `turn_3_prompt`, `turn_3_instruction_id_list`, `turn_3_kwargs`, `key`, `turn_index`, `language`
- **Description**: Multi-turn instruction following benchmark from Meta/Facebook. Contains multi-turn prompts with associated instructions and evaluation criteria across multiple languages.
- **Relevance**: Directly measures multi-turn instruction following degradation.

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/multi_if")
```

### 2. MT-Bench Prompts

- **Source**: [HuggingFaceH4/mt_bench_prompts](https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts)
- **Status**: Downloaded (full dataset)
- **Size**: ~52 KB on disk
- **Split**: train (80 examples)
- **Features**: `category`, `prompt`, `reference`, `prompt_id`
- **Description**: Multi-turn benchmark prompts from the MT-Bench evaluation suite. Contains 80 multi-turn conversation prompts spanning categories like writing, roleplay, reasoning, math, coding, extraction, STEM, and humanities.
- **Relevance**: Standard benchmark for multi-turn conversation quality evaluation.

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/mt_bench")
```

### 3. BoolQ (Google)

- **Source**: [google/boolq](https://huggingface.co/datasets/google/boolq)
- **Status**: Downloaded (full dataset)
- **Size**: ~5.1 MB on disk
- **Splits**: train (9,427 examples), validation (3,270 examples)
- **Features**: `question`, `answer`, `passage`
- **Description**: Boolean question answering dataset. Each example contains a question, a passage, and a boolean (yes/no) answer. Originally from Google Research.
- **Relevance**: Can be used to test whether multi-turn context degrades simple factual QA performance.

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/boolq")
```

### 4. LMSYS-Chat-1M

- **Source**: [lmsys/lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)
- **Status**: NOT DOWNLOADED (requires authentication)
- **Size**: Several GB (full dataset)
- **Description**: 1 million real-world conversations collected from the Chatbot Arena platform. Contains conversations with various LLMs including GPT-4, Claude, LLaMA, and others.
- **Relevance**: Real multi-turn conversations for analyzing regression patterns in the wild.

**Download instructions** (requires HuggingFace authentication):

```bash
# 1. Go to https://huggingface.co/datasets/lmsys/lmsys-chat-1m
# 2. Accept the terms of use / license agreement
# 3. Create a HuggingFace token at https://huggingface.co/settings/tokens
# 4. Set your token:
export HF_TOKEN="your_token_here"
```

```python
from datasets import load_dataset, Dataset

# Stream to avoid downloading the full dataset
ds = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)

# Take a sample of 10,000 examples
samples = list(ds.take(10000))

# Save as a HuggingFace dataset
ds_sample = Dataset.from_list(samples)
ds_sample.save_to_disk("datasets/lmsys_chat_1m")
```

### 5. WildChat (AllenAI)

- **Source**: [allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M)
- **Status**: Downloaded (10,000 example sample)
- **Size**: ~39 MB on disk (sample), full dataset is several GB
- **Split**: train (10,000 examples from ~1M total)
- **Features**: `conversation_hash`, `model`, `timestamp`, `conversation` (list of turns with content, role, language, etc.), `turn`, `language`, `openai_moderation`, `detoxify_moderation`, `toxic`, `redacted`, `state`, `country`, `hashed_ip`, `header`
- **Description**: 1 million real ChatGPT conversations collected in the wild with rich metadata including language, toxicity scores, and moderation flags.
- **Relevance**: Real multi-turn conversations for analyzing how conversation quality changes over turns.

```python
from datasets import load_from_disk
ds = load_from_disk("datasets/wildchat")

# To download the full dataset:
from datasets import load_dataset
ds_full = load_dataset("allenai/WildChat-1M")
```

## File Structure

```
datasets/
  README.md                        # This file
  .gitignore                       # Excludes large data files from git
  multi_if/
    dataset_dict.json              # HF dataset metadata
    train/                         # Arrow data files
    sample_train.json              # 5-example sample for reference
  mt_bench/
    dataset_dict.json
    train/
    sample_train.json
  boolq/
    dataset_dict.json
    train/
    validation/
    sample_train.json
    sample_validation.json
  lmsys_chat_1m/
    DOWNLOAD_INSTRUCTIONS.json     # Instructions (dataset requires auth)
  wildchat/
    data-00000-of-00001.arrow      # 10k sample data
    dataset_info.json
    state.json
    sample_train.json
```

## Loading Datasets

All successfully downloaded datasets can be loaded using the HuggingFace `datasets` library:

```python
from datasets import load_from_disk, DatasetDict

# Load individual datasets
multi_if = load_from_disk("datasets/multi_if")
mt_bench = load_from_disk("datasets/mt_bench")
boolq = load_from_disk("datasets/boolq")
wildchat = load_from_disk("datasets/wildchat")
```

## Notes

- Large data files are excluded from git via `.gitignore`. Only README, sample JSON files, and download instructions are tracked.
- Sample JSON files (`sample_*.json`) contain the first 5 examples from each split for quick reference without loading the full dataset.
- The LMSYS-Chat-1M dataset is gated and requires accepting terms on HuggingFace before download.
- WildChat contains only a 10k sample; the full 1M dataset can be downloaded separately if needed.
