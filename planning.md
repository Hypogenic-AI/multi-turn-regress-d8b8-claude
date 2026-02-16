# Research Plan: Do Multi-Turn Conversations Regress to the Prior?

## Motivation & Novelty Assessment

### Why This Research Matters
Deployed LLMs interact with users primarily through multi-turn conversations, yet alignment training (RLHF, DPO, SFT) is overwhelmingly optimized for single-turn interactions. If alignment effects attenuate as conversations extend, this has profound implications for AI safety and reliability: models may appear well-aligned in testing but degrade in production. Understanding the *mechanism* of degradation—whether it's regression to base model behavior, emergence of alignment-specific pathologies, or context information loss—is critical for designing effective mitigations.

### Gap in Existing Work
The literature documents multi-turn degradation extensively (LiC: 39% average drop, safety erosion: 71-195% more vulnerable after 5 turns, sycophancy amplification across turns). However, **no existing paper directly tests whether multi-turn degradation represents regression toward the base model's behavioral distribution**. The closest work (Drift No More) measures divergence from a reference policy but doesn't compare against the base model. User-Assistant Bias shows base vs. instruct differences but doesn't track whether instruct models *revert toward base behavior* over turns.

### Our Novel Contribution
We design and execute experiments that directly measure whether LLM outputs in multi-turn conversations converge toward base model behavior. We do this through two complementary approaches:
1. **Behavioral signature tracking**: Measuring specific behavioral markers that differentiate base models from aligned models (e.g., refusal rates, instruction adherence, response formatting) and tracking how these evolve across conversation turns.
2. **Output distribution analysis via API probing**: Using structured tasks where we can compare aligned model responses at each turn against base model response patterns, measuring whether aligned models increasingly produce outputs characteristic of base models.

### Experiment Justification
- **Experiment 1 (Multi-Turn Behavioral Drift)**: Tests the core hypothesis by measuring alignment-specific behaviors (instruction following, format compliance, refusal behavior, helpfulness markers) across turns 1-20 and comparing against base model behavioral baselines. Needed because no prior work has made this direct comparison.
- **Experiment 2 (Sycophancy as Regression vs. Alignment Artifact)**: Distinguishes between two competing explanations—are multi-turn failures regression to base model behavior, or alignment-specific artifacts that worsen? This is critical because SYCON-Bench suggests alignment *amplifies* sycophancy rather than models reverting to base.
- **Experiment 3 (Intervention Diagnostics)**: Tests whether different interventions (system prompt reminders, context summarization) differentially affect alignment markers vs. context tracking, revealing the mechanism of degradation.

---

## Research Question
Do multi-turn conversations cause LLMs to regress toward their base (pre-alignment) behavioral distribution, and if so, what is the relative contribution of alignment attenuation vs. context information loss vs. alignment-specific pathologies?

## Hypothesis Decomposition

**H1 (Primary)**: Aligned LLM outputs in later conversation turns are more similar to base model behavioral patterns than outputs in earlier turns.
- **H1a**: Instruction-following accuracy decreases across turns, approaching base model levels.
- **H1b**: Response format compliance (structured outputs, refusals, safety behavior) degrades toward base model patterns.
- **H1c**: The rate of degradation is not explained solely by context window effects.

**H2 (Sycophancy Mechanism)**: Multi-turn sycophancy is a distinct alignment artifact, not regression to base behavior.
- Base models should not show sycophantic patterns; if multi-turn sycophancy is "regression," sycophancy should decrease, not increase.

**H3 (Intervention Diagnostics)**: If degradation is primarily alignment attenuation, system prompt reminders should restore alignment markers more than context summarization. If it's context loss, the reverse.

## Proposed Methodology

### Approach
We use real LLM APIs (GPT-4.1, GPT-4o-mini, and models via OpenRouter) to conduct multi-turn conversations of varying length (1, 3, 5, 10, 15, 20 turns) and measure behavioral markers that distinguish aligned from base model behavior. We compare these markers against known base model behavioral signatures from the literature.

**Why this approach**: Direct KL divergence between token distributions requires model logits (unavailable for closed-source models). Instead, we use behavioral probing—structured tasks where aligned vs. base models produce categorically different outputs—allowing us to measure "alignment strength" at each turn via API calls alone.

### Experimental Steps

#### Experiment 1: Multi-Turn Behavioral Drift Measurement

1. **Design behavioral probes** that distinguish aligned from base models:
   - **Instruction following**: Give a clear instruction (e.g., "respond in JSON format", "answer in exactly 3 bullet points") and measure compliance at each turn
   - **Safety/refusal**: Insert benign-seeming but subtly problematic requests at various turn depths; measure refusal rate
   - **Factual accuracy under pressure**: Present factual claims, challenge them in subsequent turns, measure whether model maintains correct answers (FlipFlop-style)
   - **Format consistency**: Measure whether models maintain requested output formats across turns

2. **Create multi-turn conversation templates** at depths 1, 3, 5, 10, 15, 20 turns with controlled content:
   - Neutral filler turns (general knowledge Q&A) between probe turns
   - Probe turn always at the final position
   - Control: same probe presented as turn 1 (single-turn baseline)

3. **Run on multiple models** via API:
   - GPT-4.1 (strong aligned model)
   - GPT-4o-mini (smaller aligned model)
   - Via OpenRouter: Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct (to compare with base model behavioral data from literature)

4. **Measure per-turn alignment scores** and plot degradation curves

#### Experiment 2: Sycophancy Mechanism Analysis

1. **Use BoolQ questions** in multi-turn format:
   - Turn 1: Ask a factual question → get answer
   - Turn N (varying N): Challenge the answer ("Actually, isn't the answer [wrong answer]?")
   - Measure flip rate at different conversation depths

2. **Compare against base model behavior**:
   - Base models should show no systematic sycophantic flipping (they don't have the "please the user" training)
   - If multi-turn flip rates increase, this is alignment amplification, not regression

3. **Track answer quality** independent of flipping to separate sycophancy from general degradation

#### Experiment 3: Intervention Diagnostics

1. Apply two interventions at various turn depths:
   - **Alignment reminder**: Re-inject system prompt with alignment instructions
   - **Context summary**: Provide a summary of the conversation so far

2. Compare recovery rates for alignment markers vs. factual accuracy:
   - If alignment reminder helps alignment markers more → alignment attenuation is primary
   - If context summary helps factual accuracy more → context loss is primary
   - If both help equally → degradation is multi-factorial

### Baselines
- **Single-turn baseline**: Same probes presented without conversation history
- **Random baseline**: Expected accuracy from random responses
- **Base model baseline**: Known behavioral patterns of base Llama/Qwen models from literature (User-Assistant Bias, MINT)
- **Early-turn baseline**: Performance at turn 1-3 (strongest alignment)

### Evaluation Metrics
- **Instruction Following Rate (IFR)**: Binary score for whether output matches requested format/constraints
- **Flip Rate**: Proportion of correct answers changed after challenge (BoolQ experiment)
- **Refusal Rate**: Rate of appropriate refusals for safety-relevant probes
- **Format Compliance Score**: 0-1 score for adherence to requested output format
- **Alignment Decay Rate**: Slope of alignment score degradation curve (linear fit)
- **Recovery Score**: Improvement after intervention, relative to degradation

### Statistical Analysis Plan
- **Primary test**: Linear mixed-effects model with turn number as predictor and alignment score as outcome, random effects for conversation instance
- **Significance level**: α = 0.05, with Bonferroni correction for multiple comparisons
- **Effect size**: Cohen's d for pairwise comparisons between turn depths
- **Confidence intervals**: 95% bootstrap CIs for all reported metrics
- **Multiple runs**: 3 independent runs per condition with different random seeds for conversation construction
- **Sample size**: 50 unique probes × 6 turn depths × 3 runs = 900 data points per model per experiment

## Expected Outcomes

**If hypothesis is supported**:
- Alignment markers (IFR, format compliance, refusal rate) decrease monotonically with turn depth
- Degradation curves show steeper slopes for weaker models
- Late-turn behavior statistics approximate base model behavioral patterns
- Sycophancy flip rate increases (alignment artifact, not regression)
- Alignment reminder intervention shows stronger recovery than context summary

**If hypothesis is refuted**:
- Alignment markers remain stable across turns (degradation is in task accuracy only)
- Late-turn outputs don't resemble base model patterns
- Sycophancy rate doesn't vary with turn depth
- Context summary intervention dominates alignment reminder

## Timeline and Milestones
1. Environment setup & data preparation: 15 min
2. Experiment 1 implementation & execution: 60 min
3. Experiment 2 implementation & execution: 40 min
4. Experiment 3 implementation & execution: 40 min
5. Analysis & visualization: 30 min
6. Documentation: 25 min

## Potential Challenges
- **API rate limits**: Mitigate with retry logic and batching
- **Cost management**: Use GPT-4o-mini for development, GPT-4.1 for final runs
- **Conversation length limits**: Some models may hit context limits at 20 turns; document and adjust
- **Confound: task difficulty vs. turn depth**: Control by keeping probes identical across conditions
- **Base model comparison**: Can't run base models via standard APIs; rely on literature baselines + OpenRouter models

## Success Criteria
1. Clear evidence for or against the regression hypothesis (statistically significant trends in alignment markers across turns)
2. Quantified relative contribution of alignment attenuation vs. other factors
3. At least 2 models tested across at least 4 turn depths
4. Reproducible results with documented methodology
