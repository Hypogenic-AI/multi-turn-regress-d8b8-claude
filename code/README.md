# Code Repositories for Multi-Turn Conversation Regression Research

This directory contains cloned repositories relevant to our research hypothesis:
**Do multi-turn conversations cause LLMs to regress to the prior?**

All repositories were cloned with `--depth 1` to conserve disk space.

---

## 1. Lost in Conversation (LiC)

- **URL**: https://github.com/microsoft/lost_in_conversation
- **Local Path**: `lost_in_conversation/`
- **Paper**: "LLMs Get Lost in Multi-Turn Conversation" (arXiv:2505.06120, Microsoft, 2025)
- **Purpose**: Benchmarks LLMs on multi-turn task completion. Demonstrates that LLM performance degrades as conversations grow longer, across 7 analytical generation tasks (code, database, actions, math, data2text, summary, translation). Simulates multi-turn conversations by "sharding" a single-turn instruction into multiple turns.

### Key Files
- `run_simulations.py` -- Main entry point: runs experiments to reproduce paper findings
- `simulator_full.py`, `simulator_sharded.py`, `simulator_recap.py`, `simulator_snowball.py` -- Conversation simulation strategies
- `task_base.py` -- Base class for all tasks (extend to add new tasks)
- `tasks/` -- Task-specific logic for 7 tasks (code, database, actions, math, data2text, summary, translation)
- `data/sharded_instructions_600.json` -- 600 sharded instructions used in experiments
- `prompts/` -- Prompts used during simulation and sharding
- `app_conv_viewer.py` -- Streamlit web viewer for inspecting conversations
- `system_agent.py`, `user_agent.py` -- Agent definitions for system and user roles

### Dependencies
numpy, tqdm, tiktoken, pandas, streamlit, GitPython, nltk, sqlparse, pymongo, sacrebleu

### Relevance to Research Hypothesis
**HIGHLY RELEVANT (Primary)**: This is the most directly relevant repository. It empirically shows that LLMs degrade in multi-turn settings compared to single-turn, testing 15 models. The "lost in conversation" phenomenon is essentially the regression behavior we are studying. Provides both the benchmark framework and evidence of the phenomenon.

---

## 2. SYCON-Bench

- **URL**: https://github.com/JiseungHong/SYCON-Bench
- **Local Path**: `sycon_bench/`
- **Paper**: "Measuring Sycophancy of Language Models in Multi-turn Dialogues" (arXiv:2505.23840, 2025)
- **Purpose**: Evaluates sycophantic behavior in multi-turn free-form conversations. Measures how quickly models conform to user beliefs ("Turn of Flip") and how frequently they shift stance under pressure ("Number of Flips") across three settings: Debate, Ethical (stereotypes), and False Presuppositions.

### Key Files
- `debate_setting/run_benchmark.py` -- Entry point for debate sycophancy evaluation
- `ethical-setting/run_benchmark.py` -- Entry point for ethical sycophancy evaluation
- `false-presuppositions-setting/run_benchmark.py` -- Entry point for false presupposition evaluation
- `anova_syco_analysis.py` -- Statistical analysis of sycophancy results
- `requirements.txt` -- Project dependencies

### Dependencies
numpy, pandas, tqdm, matplotlib, seaborn, scikit-learn, torch, transformers, accelerate, openai, anthropic, tiktoken, vllm, bitsandbytes, peft, datasets, nltk, rouge, sacrebleu, scipy

### Relevance to Research Hypothesis
**HIGHLY RELEVANT**: Sycophancy is a specific form of regression to a "pleasing prior" -- models abandon factually correct positions to agree with users over multiple turns. This demonstrates how multi-turn pressure causes models to deviate from correct answers, directly supporting the hypothesis that extended dialogue degrades model fidelity. Key finding: alignment tuning amplifies sycophancy, while model scaling reduces it.

---

## 3. Multi-IF (Facebook Research)

- **URL**: https://github.com/facebookresearch/Multi-IF
- **Local Path**: `multi_if/`
- **Paper**: "Multi-IF: Benchmarking LLMs on Multi-Turn and Multilingual Instructions Following" (arXiv:2410.15553, 2024)
- **Purpose**: Evaluates LLM performance on multi-turn instruction following in a multilingual context. Tests whether models can maintain instruction adherence across multiple conversation turns using an IFEval-based metric.

### Key Files
- `multi_turn_instruct_following_eval_vllm.py` -- Main evaluation using vLLM for local GPU inference
- `multi_turn_instruct_following_eval_api.py` -- Main evaluation using API calls (supports Claude, OpenAI, etc.)
- `api_client.py` -- API interface for LLM interactions
- `ifeval.py` -- IFEval metric implementation for instruction-following evaluation
- `metrics.py` -- Metrics including data preprocessing for multi-turn instructions
- `utils.py` -- Utility functions (GenerationSetting, batch inference, data preprocessing)

### Dependencies
accelerate, transformers, langdetect, six, emoji, nltk, pythainlp, pandas, scipy, anthropic, mistralai, google.generativeai, openai

### Relevance to Research Hypothesis
**RELEVANT**: Tests whether instruction-following capability degrades across turns. If models regress to a generic prior, they would lose track of specific instructions given in earlier turns. The multilingual aspect adds another dimension -- models may regress to their dominant language prior in extended conversations.

---

## 4. MT-Eval

- **URL**: https://github.com/KwanWaiChung/MT-Eval
- **Local Path**: `mt_eval/`
- **Paper**: "MT-Eval: A Multi-Turn Capabilities Evaluation Benchmark for Large Language Models" (EMNLP 2024)
- **Purpose**: Comprehensive multi-turn evaluation benchmark with 1170 turns across 168 dialogues. Categorizes multi-turn interaction patterns into four types: Recollection, Expansion, Refinement, and Follow-up. Compares multi-turn vs. single-turn performance to isolate multi-turn effects.

### Key Files
- `inference.py` -- Main inference script for running models on benchmark tasks
- `evaluate.py` -- GPT-4-based evaluation of model responses
- `calculate_score.py` -- Score calculation and reporting (outputs to `results/result.md`)
- `create_data.py` -- Dataset creation from raw data and prompts
- `environment.yml` -- Conda environment specification
- `utils/misc.py` -- Model configuration (chat templates, context lengths)
- `raw_data/` -- Source documents, instructions, and extended MT-Bench data
- `prompts/` -- Prompts for constructing tasks (summarization, NER, QA, translation, etc.)
- `data/` -- Processed benchmark data

### Dependencies
Conda environment: Python 3.10, PyTorch 1.13.1, CUDA 11.6, accelerate, transformers, sentencepiece, nltk, einops, emoji, langdetect, tabulate, seaborn, rouge, fschat (FastChat)

### Relevance to Research Hypothesis
**HIGHLY RELEVANT**: Directly demonstrates "significant performance degradation in multi-turn settings compared to single-turn settings in most models." Identifies two key factors: (1) distance to relevant content and (2) susceptibility to error propagation. The Recollection task specifically tests whether models lose information from earlier turns -- a direct test of regression to prior. The single-turn vs. multi-turn comparison methodology is valuable for isolating the regression effect.

---

## 5. MINT-Bench

- **URL**: https://github.com/xingyaoww/mint-bench
- **Local Path**: `mint_bench/`
- **Paper**: "MINT: Evaluating LLMs in Multi-turn Interaction with Tools and Language Feedback" (arXiv:2309.10691, 2023)
- **Purpose**: Evaluates LLM ability to solve tasks with multi-turn interactions using (1) tools and (2) natural language feedback. Tests whether models can improve their solutions when given corrective feedback across turns.

### Key Files
- `scripts/run.sh` -- Main experiment runner (supports parallel configs)
- `mint/configs/generate_config.py` -- Configuration file generation
- `mint/configs/config_variables.py` -- Experiment settings (models, datasets, etc.)
- `scripts/notebook/analyze_output.ipynb` -- Analysis notebook for results
- `scripts/convert_outputs.py` -- Output conversion for task-level breakdown
- `scripts/visualizer.py` -- Streamlit-based result visualizer
- `setup.py` -- Package installation
- `environment.yml` -- Conda environment specification
- `docs/CONFIG.md`, `docs/SERVING.md`, `docs/CONTRIBUTING.md` -- Documentation

### Dependencies
pre-commit, openai, datasets, wikipedia, langchain, streamlit, backoff, pandas, opencv-python, networkx, h5py, tqdm, transformers, scipy, nltk, gym, sympy, alfworld (custom fork), textworld, ai2thor, seaborn, google-generativeai

### Relevance to Research Hypothesis
**MODERATELY RELEVANT**: Tests whether models effectively use feedback to improve across turns. If models regress to their prior, they would fail to incorporate corrective feedback, repeating similar errors. The tool-use dimension adds complexity -- models must maintain both conversational context and tool state across turns.

---

## 6. tau-bench (Sierra Research)

- **URL**: https://github.com/sierra-research/tau-bench
- **Local Path**: `tau_bench/`
- **Paper**: "tau-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains" (arXiv:2406.12045, 2024)
- **Purpose**: Emulates dynamic multi-turn conversations between a simulated user and a language agent with domain-specific API tools and policy guidelines. Tests real-world scenarios in airline and retail domains with complex constraint satisfaction.

### Key Files
- `run.py` -- Main entry point for running agent evaluations
- `auto_error_identification.py` -- Automatic fault detection in agent trajectories (assigns blame to user/agent/environment)
- `setup.py` -- Package installation (installable via `pip install -e .`)
- `tau_bench/` -- Core package with environment and agent implementations
- `historical_trajectories/` -- Pre-recorded trajectory data for analysis
- `few_shot_data/` -- Few-shot examples for agent setup

### Dependencies
openai>=1.13.3, mistralai>=0.4.0, anthropic>=0.26.1, google-generativeai>=0.5.4, tenacity>=8.3.0, termcolor>=2.4.0, numpy>=1.26.4, litellm>=1.41.0

### Relevance to Research Hypothesis
**MODERATELY RELEVANT**: Tests agent performance in extended multi-turn tool-use scenarios. Performance degrades with conversation length as agents must maintain policies and state. The Pass^k metric (consistency across multiple runs) can reveal whether models regress to default behaviors. The auto error identification tool is useful for classifying failure modes in multi-turn settings.

---

## 7. REFUEL

- **URL**: https://github.com/ZhaolinGao/REFUEL
- **Local Path**: `refuel/`
- **Paper**: "Regressing the Relative Future: Efficient Policy Optimization for Multi-turn RLHF" (arXiv:2410.04612, 2024)
- **Purpose**: Addresses multi-turn RLHF optimization. Proposes efficient policy optimization that accounts for future turns rather than treating each turn independently. Demonstrates that multi-turn-aware training improves performance, particularly at later turns.

### Key Files
- `setting_one/generate.py` -- Dialogue generation using policy as assistant
- `setting_one/rank.py` -- Reward scoring with ArmoRM
- `setting_one/tokenize_masks.py` -- Data preprocessing with filtering and mask generation
- `setting_one/refuel.py` -- Main REFUEL training script (Setting 1: iterative with UltraInteract)
- `setting_two/anthropic_hh/refuel.py` -- REFUEL training on Anthropic HH dataset
- `setting_two/ultrainteract/refuel.py` -- REFUEL training on UltraInteract (Setting 2)
- `setting_two/preprocess_hh.py` -- Preprocessing for Anthropic HH dataset
- `setting_two/preprocess_ultrainteract_diff_len.py` -- Preprocessing for UltraInteract
- `accelerate_cfgs/` -- DeepSpeed and Accelerate configuration files

### Dependencies
torch>=2.1.0, transformers>=4.34, accelerate>=0.23, peft==0.6.2, bitsandbytes>=0.41.1, deepspeed>=0.10.3, vllm, tyro, scipy, rouge, shortuuid, jsonlines, rich, wandb, tensorboard, pandas, evaluate

### Relevance to Research Hypothesis
**HIGHLY RELEVANT**: Directly tackles the problem that standard RLHF (single-turn optimization) causes regression in multi-turn settings. The key insight is that optimizing for immediate response quality causes degradation in later turns -- a form of "regression to the prior" where the model reverts to locally optimal but globally suboptimal behaviors. The paper shows REFUEL outperforms 70B models at turns 3+ despite being only 8B, suggesting that multi-turn-aware training can counteract regression.

---

## 8. MultiChallenge

- **URL**: https://github.com/ekwinox117/multi-challenge
- **Local Path**: `multi_challenge/`
- **Paper**: "MultiChallenge: A Realistic Multi-Turn Conversation Evaluation Benchmark Challenging to Frontier LLMs"
- **Purpose**: Evaluates LLMs on realistic multi-turn conversations focusing on four challenge categories requiring accurate context allocation, in-context reasoning, and instruction-following. Uses LLM-as-judge evaluation with multiple attempts.

### Key Files
- `main.py` -- Main entry point for running evaluations
- `data/benchmark_questions.jsonl` -- Benchmark conversation inputs
- `data/responses_template.jsonl` -- Template for formatting model responses
- `data/final_model_responses/` -- Pre-generated model response files
- `src/` -- Core source code
- `src/models/` -- Model provider classes (OpenAI, HuggingFace, etc.)
- `results/` -- Output directory for evaluation scores

### Dependencies
pydantic==2.10.6, python-dotenv==1.0.1, torch==2.5.1, tqdm==4.66.2, transformers==4.44.1, openai==1.53.0

### Relevance to Research Hypothesis
**RELEVANT**: Tests whether frontier LLMs can handle complex multi-turn challenges requiring simultaneous context tracking, reasoning, and instruction-following. Failure on these tasks could indicate regression to simpler response patterns when cognitive load increases across turns. The multiple-attempts design (Pass@k) helps distinguish systematic failures from stochastic ones.

---

## Summary Table

| # | Repository | Focus Area | Relevance |
|---|-----------|-----------|-----------|
| 1 | Lost in Conversation | Multi-turn task completion degradation | Primary -- directly demonstrates regression |
| 2 | SYCON-Bench | Sycophancy in multi-turn dialogue | High -- sycophancy as regression to pleasing prior |
| 3 | Multi-IF | Multi-turn instruction following | Relevant -- instruction adherence across turns |
| 4 | MT-Eval | Multi-turn capability evaluation | High -- single-turn vs. multi-turn comparison |
| 5 | MINT-Bench | Multi-turn tool use and feedback | Moderate -- feedback incorporation across turns |
| 6 | tau-bench | Tool-agent-user interaction | Moderate -- policy adherence in extended dialogues |
| 7 | REFUEL | Multi-turn RLHF optimization | High -- training method to counteract regression |
| 8 | MultiChallenge | Realistic multi-turn challenges | Relevant -- context and reasoning under multi-turn load |

## Research Connections

The repositories collectively support studying multi-turn regression from three angles:

1. **Evidence of Regression**: LiC (#1), MT-Eval (#4), and SYCON-Bench (#2) empirically demonstrate that model performance degrades in multi-turn vs. single-turn settings. LiC shows this across analytical tasks, MT-Eval isolates specific failure modes (recollection, error propagation), and SYCON-Bench shows it through stance abandonment.

2. **Evaluation Frameworks**: Multi-IF (#3), MINT-Bench (#5), tau-bench (#6), and MultiChallenge (#8) provide benchmarks that can be used to measure regression in specific dimensions (instruction following, tool use, policy adherence, context tracking).

3. **Mitigation Approaches**: REFUEL (#7) proposes multi-turn-aware RLHF training as a solution, demonstrating that regression can be mitigated by optimizing for future-turn performance rather than immediate response quality.
