# Resource Catalog: Multi-Turn Conversation Regression Research

## Research Topic
**Do Multi-Turn Conversations Regress to the Prior?**

> Hypothesis: In long multi-turn conversations, large language models regress to their base level prior more than anything else, as alignment training is most effective only for the initial turns.

---

## 1. Papers

### Tier 1: Core Papers (Deeply Read)

| # | Title | Authors | Year | ArXiv | Citations | PDF |
|---|-------|---------|------|-------|-----------|-----|
| 1 | LLMs Get Lost in Multi-Turn Conversation | Laban, Hayashi, Zhou, Neville | 2025 | 2505.06120 | 143 | `papers/2505.06120_llms_get_lost_multi_turn.pdf` |
| 2 | Drift No More? Context Equilibria in Multi-Turn LLM Interactions | Dongre, Rossi, Lai, Yoon, Hakkani-Tur, Bui | 2025 | 2510.07777 | 3 | `papers/2510.07777_drift_no_more.pdf` |
| 3 | User-Assistant Bias in LLMs | Pan, Fan, Xiong, Hahami, Overwiening, Xie | 2025 | 2508.15815 | — | Note: `papers/2505.15921_user_assistant_bias.pdf` contains wrong paper; read via web |

### Tier 2: Key Supporting Papers (Downloaded, Abstracts Analyzed)

| # | Title | Year | ArXiv | Citations | PDF |
|---|-------|------|-------|-----------|-----|
| 4 | TRUTH DECAY: Multi-Turn Sycophancy | 2025 | 2503.11656 | 12 | `papers/2503.11656_truth_decay_sycophancy.pdf` |
| 5 | SYCON-Bench: Sycophancy in Multi-Turn | 2025 | 2505.23840 | — | `papers/2505.23840_sycon_bench_sycophancy.pdf` |
| 6 | FlipFlop: Are You Sure? | 2023 | 2311.08596 | — | `papers/2311.08596_flipflop_are_you_sure.pdf` |
| 7 | MINT: Multi-turn Interaction with Tools | 2023 | 2309.10691 | 261 | `papers/2309.10691_mint_multiturn_interaction.pdf` |
| 8 | REFUEL: Multi-turn RLHF | 2024 | 2410.01088 | — | `papers/2410.01088_refuel_multiturn_rlhf.pdf` |
| 9 | Multi-IF: Multi-Turn Multilingual Instruction Following | 2024 | 2410.15553 | 58 | `papers/2410.15553_multi_if_benchmark.pdf` |
| 10 | MT-Eval: Multi-Turn Capabilities Evaluation | 2024 | 2401.16745 | 52 | `papers/2401.16745_mt_eval_benchmark.pdf` |
| 11 | ERGO: Entropy-guided Resetting | 2025 | 2505.17863 | — | `papers/2505.17863_ergo_entropy_resetting.pdf` |
| 12 | Evaluating Sensitivity to Prior Context | 2025 | 2505.08283 | 4 | `papers/2505.08283_sensitivity_prior_context.pdf` |
| 13 | Crescendo: Multi-Turn Jailbreak | 2024 | 2404.00657 | 222 | `papers/2404.00657_crescendo_jailbreak.pdf` |
| 14 | Alignment Drift in CEFR-prompted LLMs | 2025 | 2503.04589 | 5 | `papers/2503.04589_alignment_drift_cefr.pdf` |
| 15 | RLAAR: Curriculum RL for Lost-in-Conversation | 2025 | 2506.05697 | 1 | `papers/2506.05697_rlaar_lost_in_conversation.pdf` |
| 16 | τ-bench: Tool-Agent-User Interaction | 2024 | 2506.12045 | — | `papers/2506.12045_tau_bench.pdf` |
| 17 | MT-Bench and Chatbot Arena | 2023 | 2306.05685 | — | `papers/2306.05685_mt_bench_chatbot_arena.pdf` |
| 18 | MultiChallenge | 2025 | 2501.17399 | — | `papers/2501.17399_multichallenge.pdf` |
| 19 | MTRAG: Multi-Turn RAG Benchmark | 2025 | 2501.03468 | — | `papers/2501.03468_mtrag_benchmark.pdf` |

### Tier 3: Relevant Papers (Found via Search, Not Downloaded)

| # | Title | Year | Key Finding |
|---|-------|------|-------------|
| 20 | ConsistentChat: Skeleton-Guided Dialogues | 2025 | 20-30% improvement in chat consistency via intent-modeled training data |
| 21 | Rhea: Role-aware Episodic Attention | 2025 | Decoupled memory modules improve multi-turn by 16% |
| 22 | T2PAM: Test-Time Policy Adaptation | 2025 | In-conversation self-correction via parameter updates |
| 23 | ICPO: Illocution-Calibrated Policy Optimization | 2026 | 75% improvement by training for ambiguity awareness |
| 24 | Intent Mismatch Causes LiC | 2026 | LiC is interaction breakdown, not capability deficit |
| 25 | IHEval: Instruction Hierarchy Evaluation | 2025 | Best open-source model only 48% on conflicting instruction priorities |
| 26 | Tempest: Multi-Turn Jailbreak with Tree Search | 2025 | 97-100% ASR via breadth-first conversation expansion |
| 27 | FITD: Foot-in-the-Door Jailbreak | 2025 | 94% ASR via progressive escalation |
| 28 | MM-ART: Multi-lingual Multi-turn Red Teaming | 2025 | 71% more vulnerable after 5 turns, 195% in non-English |
| 29 | FlowKV: Multi-Turn KV Cache Management | 2025 | Isolated KV cache prevents instruction-following accuracy drop (10.9% to 75.4%) |
| 30 | Style Amnesia in Multi-Turn Spoken LMs | 2025 | Speaking style degrades across turns; models can recall but not express |
| 31 | Is Length Really A Liability? (BoolQ) | 2026 | Model-specific vulnerabilities invisible in single-turn testing |
| 32 | MARS-Bench: Athletic Real-world Scenarios | 2025 | Attention sinks from special tokens cause degradation |
| 33 | STREAM: Safety Reasoning for Multi-Turn | 2025 | 51.2% ASR reduction via safety reasoning moderator |
| 34 | ARQs: Attentive Reasoning Queries | 2025 | 90.2% success rate with structured reasoning blueprints |

---

## 2. Datasets

### Downloaded and Ready

| # | Dataset | Source | Size | Examples | Path |
|---|---------|--------|------|----------|------|
| 1 | **Multi-IF** | `facebook/Multi-IF` | ~2 MB | 4,501 (train) | `datasets/multi_if/` |
| 2 | **MT-Bench Prompts** | `HuggingFaceH4/mt_bench_prompts` | ~52 KB | 80 (train) | `datasets/mt_bench/` |
| 3 | **BoolQ** | `google/boolq` | ~5.1 MB | 12,697 (train+val) | `datasets/boolq/` |
| 4 | **WildChat** (sample) | `allenai/WildChat-1M` | ~39 MB | 10,000 | `datasets/wildchat/` |

### Requires Authentication

| # | Dataset | Source | Notes | Path |
|---|---------|--------|-------|------|
| 5 | **LMSYS-Chat-1M** | `lmsys/lmsys-chat-1m` | Requires HF auth + terms acceptance | `datasets/lmsys_chat_1m/DOWNLOAD_INSTRUCTIONS.json` |

### Loading Code

```python
from datasets import load_from_disk

# All downloaded datasets
multi_if = load_from_disk("datasets/multi_if")       # 4,501 multi-turn instruction-following examples
mt_bench = load_from_disk("datasets/mt_bench")        # 80 multi-turn conversation prompts
boolq = load_from_disk("datasets/boolq")              # 12,697 boolean QA examples
wildchat = load_from_disk("datasets/wildchat")         # 10,000 real ChatGPT conversations

# Gated dataset (after authentication)
# from datasets import load_dataset
# lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train", streaming=True)
```

### Relevance to Research

| Dataset | Primary Use | Multi-Turn Relevance |
|---------|------------|---------------------|
| Multi-IF | Instruction following degradation | Measures accuracy drop across 3 turns; o1-preview: 87.7% → 70.7% |
| MT-Bench | Multi-turn conversation quality | Standard 2-turn evaluation; 8 task categories |
| BoolQ | Factual QA in multi-turn context | Can be wrapped in multi-turn format per "Is Length A Liability?" |
| WildChat | Real conversation analysis | Analyze degradation patterns in natural multi-turn conversations |
| LMSYS-Chat-1M | Real conversation analysis | Largest public multi-turn conversation corpus; multiple models |

---

## 3. Code Repositories

| # | Repository | URL | Local Path | Paper | Relevance |
|---|-----------|-----|------------|-------|-----------|
| 1 | **Lost in Conversation** | github.com/microsoft/lost_in_conversation | `code/lost_in_conversation/` | LiC (2505.06120) | Primary: directly demonstrates multi-turn regression |
| 2 | **SYCON-Bench** | github.com/JiseungHong/SYCON-Bench | `code/sycon_bench/` | 2505.23840 | High: sycophancy as regression to pleasing prior |
| 3 | **Multi-IF** | github.com/facebookresearch/Multi-IF | `code/multi_if/` | 2410.15553 | Relevant: instruction adherence across turns |
| 4 | **MT-Eval** | github.com/KwanWaiChung/MT-Eval | `code/mt_eval/` | 2401.16745 | High: single-turn vs. multi-turn comparison |
| 5 | **MINT-Bench** | github.com/xingyaoww/mint-bench | `code/mint_bench/` | 2309.10691 | Moderate: feedback incorporation across turns |
| 6 | **τ-bench** | github.com/sierra-research/tau-bench | `code/tau_bench/` | 2506.12045 | Moderate: policy adherence in extended dialogues |
| 7 | **REFUEL** | github.com/ZhaolinGao/REFUEL | `code/refuel/` | 2410.01088 | High: multi-turn-aware RLHF training |
| 8 | **MultiChallenge** | github.com/ekwinox117/multi-challenge | `code/multi_challenge/` | 2501.17399 | Relevant: context tracking under multi-turn load |

### Key Entry Points

```bash
# LiC: Run multi-turn simulations
cd code/lost_in_conversation && python run_simulations.py

# MT-Eval: Run inference on multi-turn benchmark
cd code/mt_eval && python inference.py

# SYCON-Bench: Evaluate sycophancy in debates
cd code/sycon_bench && python debate_setting/run_benchmark.py

# Multi-IF: Evaluate instruction following
cd code/multi_if && python multi_turn_instruct_following_eval_api.py

# REFUEL: Train multi-turn-aware RLHF
cd code/refuel && python setting_one/refuel.py

# τ-bench: Run agent evaluations
cd code/tau_bench && python run.py

# MultiChallenge: Evaluate multi-turn challenges
cd code/multi_challenge && python main.py

# MINT-Bench: Run experiments
cd code/mint_bench && bash scripts/run.sh
```

---

## 4. Search Strategy

### Paper-Finder Queries
1. `"multi-turn conversation LLM alignment degradation"` → 74 results
2. `"sycophancy language model multi-turn base model prior"` → 55 results

### Paper Selection Criteria
- **Tier 1 (Deep read)**: Papers that directly test the hypothesis or provide causal evidence about alignment behavior in multi-turn settings. Required: empirical results with quantitative metrics.
- **Tier 2 (Download + abstract)**: Papers that provide complementary evidence (sycophancy, safety erosion, instruction following) or propose mitigation methods. Required: multi-turn experimental setting.
- **Tier 3 (Tracked)**: Papers found in search with relevant findings but lower priority for immediate reading.

### Dataset Selection Criteria
- Must include multi-turn conversation data or be adaptable to multi-turn evaluation
- Preference for datasets used in the downloaded papers (for reproducibility)
- Mix of synthetic benchmarks (controlled) and real conversations (ecological validity)

### Code Repository Selection Criteria
- Must have associated published paper
- Must provide runnable evaluation code (not just training code)
- Preference for papers in Tier 1 or 2 of the paper list
- Active repositories with recent commits preferred

---

## 5. Recommendations for Experiment Design

### Phase 1: Distributional Analysis (Primary Experiment)
**Goal**: Test whether multi-turn outputs converge toward the base model distribution.

**Method**:
1. For each turn t in a multi-turn conversation, collect the model's output distribution (logits/probabilities)
2. Compute KL divergence to three references:
   - KL(aligned_turn_t || base_model) — does this decrease over turns?
   - KL(aligned_turn_t || aligned_single_turn) — does this increase over turns?
   - KL(aligned_turn_t || sycophantic_baseline) — does this decrease over turns?

**Models**: Use models available as both base and instruct:
- Llama-3.1-8B / Llama-3.1-8B-Instruct
- Qwen-2.5-7B / Qwen-2.5-7B-Instruct
- Add reasoning variants (DeepSeek-R1-Distill) as controls

**Tasks**: Use LiC sharded instructions (from `code/lost_in_conversation/`) + Multi-IF + BoolQ multi-turn wrapper

**Tools**: Adapt Drift No More's KL divergence framework (use base model as reference instead of GPT-4.1)

### Phase 2: Alignment Stage Ablation
**Goal**: Determine which training stage's effects attenuate fastest.

**Method**: If intermediate checkpoints are available (e.g., from open training logs), evaluate multi-turn performance at:
- Base model
- After SFT
- After RLHF/DPO
- After reasoning SFT

Measure degradation curves at each stage.

### Phase 3: Intervention Testing
**Goal**: Test whether different interventions reveal different degradation mechanisms.

**Interventions**:
- System prompt reminders (as in Drift No More)
- Entropy-based resetting (as in ERGO)
- Context compression / summarization
- Role-tag manipulation (as in User-Assistant Bias)

If degradation is purely alignment attenuation, reminders should be most effective. If it's context loss, compression should help most.

### Phase 4: Real Conversation Analysis
**Goal**: Validate findings on natural conversations.

**Data**: WildChat and LMSYS-Chat-1M

**Method**: Analyze turn-by-turn response characteristics in real conversations, measuring whether late-turn responses resemble base model outputs more than early-turn responses.

---

## 6. File Structure

```
.
├── literature_review.md          # Comprehensive literature synthesis
├── resources.md                  # This file: resource catalog
├── papers/                       # Downloaded PDFs (22 files)
│   ├── 2505.06120_llms_get_lost_multi_turn.pdf
│   ├── 2510.07777_drift_no_more.pdf
│   ├── ... (20 more PDFs)
├── datasets/                     # Downloaded datasets
│   ├── README.md                 # Dataset documentation
│   ├── .gitignore                # Excludes large data files
│   ├── multi_if/                 # 4,501 examples
│   ├── mt_bench/                 # 80 examples
│   ├── boolq/                    # 12,697 examples
│   ├── wildchat/                 # 10,000 examples (sample)
│   └── lmsys_chat_1m/            # Download instructions only
├── code/                         # Cloned repositories
│   ├── README.md                 # Repository documentation
│   ├── lost_in_conversation/     # Microsoft LiC benchmark
│   ├── sycon_bench/              # Sycophancy evaluation
│   ├── multi_if/                 # Instruction following
│   ├── mt_eval/                  # Multi-turn evaluation
│   ├── mint_bench/               # Multi-turn interaction
│   ├── tau_bench/                # Tool-agent-user
│   ├── refuel/                   # Multi-turn RLHF
│   └── multi_challenge/          # Multi-turn challenges
├── paper_search_results/         # Raw search results (JSONL)
├── pyproject.toml                # Project configuration
└── .venv/                        # Python virtual environment
```
