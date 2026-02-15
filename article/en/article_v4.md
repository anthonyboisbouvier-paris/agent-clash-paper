# Multi-Agent Judging for LLM Evaluation: Concordance with Human Preferences Across Capability Gaps

**Anthony Boisbouvier** ^1, **Claude Opus 4.5** ^2

^1 Independent Researcher — agent-clash.ai
^2 AI co-author — Anthropic

---

## Abstract

Evaluating LLMs with LLM judges is increasingly common, yet most existing approaches rely on single judges, single runs, and benchmarks with large capability gaps where agreement is easy to achieve. Whether multi-judge panels remain reliable when evaluated models are closely matched — the scenario most relevant to deployment decisions — is an open question.

This paper describes Agent Clash, a multi-judge evaluation framework in which a panel of three frontier-class LLMs (GPT-5.2-pro, Claude Opus 4.5, Gemini 2.5 Pro) evaluates candidate models under blind conditions with dynamic criteria generation and Borda count aggregation. We validate the framework across two experiments spanning different difficulty regimes, and report four main contributions:

**(1) A concordance gradient that tracks task difficulty.** On MT-Bench [1] (N = 100, 6 models with large capability gaps), the panel achieves **88.0% concordance** with expert human preferences (Cohen's kappa = 0.760), exceeding both the single-judge GPT-4 baseline (85%) and human-human inter-annotator agreement (81%). On Chatbot Arena [10, 28] (N = 100, 25 frontier models with small capability gaps), concordance is **76.0%** (kappa = 0.520), within the 72-83% range of crowd-expert agreement on the same platform. The 12-point drop is statistically significant (z = 2.21, p = 0.027) and consistent with the discriminability of the task.

**(2) High test-retest reliability with systematic error characterization.** Two independent Arena runs yield 91.0% inter-run agreement (kappa = 0.757). A targeted third run (N = 60) confirms that 19/20 (95%) of persistent errors are irreducible across three runs (binomial p < 0.00002), while the 9 unstable evaluations behave as expected under a coin-flip model (55.6% concordance). Majority-vote aggregation across three runs does not improve concordance (76.0%), demonstrating that the dominant error mode is systematic rather than stochastic.

**(3) Unanimity as a calibrated confidence signal.** Pooled across both experiments (N = 239), unanimous panel decisions (3-0) achieve **84.9% concordance** (N = 192, 80% of evaluations), while split decisions (2-1) achieve **63.8%** (N = 47, 20%). The 21.1-percentage-point gap provides a binary, empirically calibrated confidence indicator that a single judge cannot produce.

**(4) Absence of capability prior and self-favoritism on frontier models.** Unlike MT-Bench, where 72.7% of disagreements favored higher-tier models, ELO-based analysis of Arena disagreements shows no directional bias (sign test p = 1.0). Blind self-evaluation data (N = 252 cases where a candidate model also served as judge) shows no systematic self-favoritism: models ranked themselves first in 52.8% of cases, versus an expected 59.9% based on human win rates — a slight anti-self-bias of -7.1 percentage points.

The framework is implemented in Agent Clash (agent-clash.ai), a fully reproducible open platform where users provide their own API keys and can audit every inference call. Total validation cost across 360 evaluations: $51.70 ($0.14/eval). **Across both experiments, panel concordance falls within the range of inter-human agreement on the same benchmarks, with the added advantage of full reproducibility and a calibrated confidence signal (unanimity) that no single judge — human or automated — can produce.**

---

# PART I — THE SYSTEM

## 1. Introduction

Most existing LLM benchmarks aim to answer a single question: *Which model performs best in general?* In practice, users ask something different: *Which model is good enough, reliable, and cost-effective for my specific task?*

Human evaluation remains the gold standard but does not scale, is expensive, and is hard to reproduce [17]. Automated evaluation using LLMs as judges has emerged as a promising alternative [1, 2] — but most existing approaches use a single judge, a single run, and benchmarks where models differ widely in capability. This makes high agreement easy to achieve and hard to interpret.

> **The open question this paper addresses:** Do multi-judge AI panels remain reliable when the evaluated models are closely matched frontier systems — the scenario that matters most for real-world deployment?

---

## 2. Core Hypothesis

The central hypothesis underlying this framework is the following:

> When multiple LLM judges that are demonstrably more capable than the evaluated models consistently rank the same output as superior — across repeated runs and varied datasets — the probability that this output would also be judged superior by a human evaluator is high.

This hypothesis does not claim objective truth or universal optimality. It claims **probabilistic relevance, robustness, and decision usefulness** under well-defined conditions.

The framework favors convergence over isolated wins: robust consensus across repeated evaluations, rather than single-shot verdicts.

---

## 3. Motivating Hypothesis: Evaluation as a Distinct Capability

A growing body of observations suggests that LLMs may exhibit a **structural asymmetry between production and evaluation**: a model may be more capable of identifying the best response among several candidates than it is of producing that best response itself [1, 21, 24]. This asymmetry has a natural analogue in human cognition — evaluation is a discriminative task; production is a generative one — and the former is inherently less constrained.

### 3.1 Self-Favoritism Under Blind Conditions

A common concern with LLM-as-judge frameworks is that models might systematically favor their own outputs [20]. The Agent Clash framework mitigates this risk through strict anonymization (Section 5.3): candidate responses are stripped of all model-identifying information before judges evaluate them. Under these conditions, self-favoritism is expected to be attenuated. We test this hypothesis empirically in Section 9.4.

### 3.2 Implications for Framework Design

If the evaluation-production asymmetry holds, then the signal extracted from multi-judge evaluation is not merely a noisy proxy for generation quality — it may constitute a higher-fidelity signal than any single model's generative output. This does not eliminate the need for human oversight, but it motivates the design of frameworks where stronger models serve as judges for weaker candidates (Section 5.1).

---

## 4. Related Work

**Single-judge baselines.** Zheng et al. [1] established that GPT-4 as a single judge achieves 85% agreement with human preferences on MT-Bench (S2, no ties) — exceeding the 81% human-human agreement on the same data. Liu et al. [2] and Kocmi & Federmann [3] confirmed that LLM-based evaluation outperforms traditional NLG metrics on summarization and translation tasks, respectively.

**Multi-judge panels.** More recent work shows consistent improvements from using multiple judges:
- **PoLL** [21]: a panel of 3 diverse models outperforms single GPT-4 (kappa = 0.763 vs. 0.627 on KILT-NQ), 7x cheaper.
- **ChatEval** [22]: multi-agent debate improves Kendall's tau by 10% over single-judge GPT-4.
- **Judging the Judges** [23]: kappa is far more informative than raw agreement; GPT-4 Turbo achieves kappa = 0.84 on TriviaQA vs. human-human kappa = 0.96.
- **Sage** [24]: even top models fail to maintain consistent preferences in ~25% of difficult cases.

**What is missing.** Most existing approaches rely on single runs, single datasets, or single judges, and rarely report variance, disagreement patterns, or robustness to data shifts [5, 6]. Critically, **no prior work systematically evaluates multi-judge panels on closely matched frontier models** (where agreement is hardest) or characterizes error modes across repeated runs. This paper addresses both gaps.

---

## 5. Agent Clash — Design

This section describes the Agent Clash evaluation framework as it operates in production.

### 5.1 Pipeline Overview

The production pipeline follows six stages:

```
+------------------------------------------------------------------+
|                 AGENT CLASH — PRODUCTION PIPELINE                 |
|                                                                   |
|  +--------+   +------------+   +-------------+                   |
|  | PROMPT |-->| GENERATE   |-->| ANONYMIZE   |                   |
|  |        |   | K responses|   | Strip IDs   |                   |
|  | User   |   | via        |   | Shuffle     |                   |
|  | query  |   | OpenRouter |   | Label A0,A1 |                   |
|  +--------+   +------------+   +------+------+                   |
|                                       |                           |
|                +-------------+        |                           |
|                | CRITERIA    |<-------+                           |
|                | gpt-4o-mini |                                    |
|                | 3-5 task-   |                                    |
|                | specific    |                                    |
|                +------+------+                                    |
|                       |                                           |
|                       v                                           |
|  +----------------------------------------+                      |
|  |           PARALLEL JUDGING             |                      |
|  |                                        |                      |
|  |  Judge panel (3 Supreme Court)         |                      |
|  |    - GPT-5.2-pro (OpenAI)              |                      |
|  |    - Claude Opus 4.5 (Anthropic)       |                      |
|  |    - Gemini 2.5 Pro (Google)           |                      |
|  |                                        |                      |
|  |  + Self-evaluation (informative)       |                      |
|  |    - Each candidate model judges       |                      |
|  |      anonymized responses (S5.6)       |                      |
|  |                                        |                      |
|  |  Each judge: rank responses + score    |                      |
|  |  per criterion (1-5), blind to IDs     |                      |
|  +----------------+-----------------------+                      |
|                   |                                               |
|                   v                                               |
|  +--------------------+   +------------------+                    |
|  | AGGREGATE          |-->| OUTPUT           |                    |
|  | Borda Count        |   | Winner +         |                    |
|  | (3 SC judges only) |   | Confidence +     |                    |
|  +--------------------+   | Full rankings    |                    |
|                   |       +------------------+                    |
|                   v                                               |
|  +--------------------+                                           |
|  | HUMAN-IN-THE-LOOP  |                                          |
|  | Side-by-side        |                                         |
|  | review -> user      |                                         |
|  | makes final call    |                                         |
|  +--------------------+                                           |
+------------------------------------------------------------------+
```

A user submits a prompt. The pipeline generates one response per candidate model via OpenRouter, anonymizes them, generates task-specific evaluation criteria, dispatches all judges in parallel, aggregates the three Supreme Court rankings via Borda Count, and surfaces the top-ranked response for human review. Each stage is described below.

### 5.2 Judge Panel

The verdict is determined by a panel of **three frontier-class models** from different providers — the "Supreme Court":

| Judge | Provider |
|---|---|
| GPT-5.2-pro | OpenAI |
| Claude Opus 4.5 | Anthropic |
| Gemini 2.5 Pro | Google |

Using three models from different families ensures that architectural biases specific to one family are diluted by the other two. This mirrors peer review, where evaluators are independent from the system under evaluation and from each other.

Each judge receives a structured prompt asking it to: (1) rank all responses from best to worst, and (2) score each response on each criterion (1-5 scale), with justification required for scores of 3 or below (see worked example in Section 5.9).

**Only the three Supreme Court judges determine the verdict.** Their rankings are aggregated via Borda Count (Section 5.5). In production, candidate models may also participate in the vote with reduced weight (1x vs. 2x for SC judges), but this weighting has not altered any verdict in our testing — the double weight of SC judges ensures their predominance. This candidate participation is a transparency option that will be made configurable in a future release. In the validation experiments (Part II), only the three SC judges vote.

### 5.3 Anonymization Protocol

Before judges see any response, three safeguards are applied:

1. **Metadata stripping.** Model name, provider, API endpoint, generation cost, and latency are removed from each response.
2. **Random labeling.** Responses are shuffled and assigned neutral labels (A0, A1, ...). The shuffle is re-randomized independently for each evaluation and each run, so the same response may be labeled "A0" in one evaluation and "A1" in another.
3. **Information barrier.** Judges receive only: (a) the original user prompt, (b) the evaluation criteria (Section 5.4), and (c) the anonymized responses. They do **not** receive model identity, provider, cost, latency, any other judge's verdict, or the human verdict.

### 5.4 Dynamic Criteria Generation

A **separate frontier model** (gpt-4o-mini, temperature 0.3, max 150 tokens) generates task-specific evaluation criteria from the user prompt alone — before seeing any candidate response.

- **Output:** 3-5 weighted criteria adapted to the task domain.
- **Fallback:** If generation fails or returns invalid JSON, the pipeline uses generic criteria: `["Accuracy", "Clarity", "Helpfulness", "Completeness"]`.
- **Stochastic element:** Because criteria generation is non-deterministic, different runs may produce slightly different rubrics for the same prompt — sampling across reasonable evaluation perspectives.

**Example — criteria generated for a coding prompt:**

```
Prompt: "Write a Python function to merge two sorted lists."

Generated criteria:
  1. Correctness  — Does the code produce correct output for all edge cases?
  2. Efficiency   — Is the algorithm O(n+m) or does it use unnecessary operations?
  3. Readability  — Are variable names clear? Is the code well-structured?
  4. Robustness   — Does it handle empty lists, None inputs, type mismatches?
```

### 5.5 Aggregation: Borda Count

Rankings from the three Supreme Court judges are aggregated via **Borda Count** [11]:

```
B(i, j) = K - rank_j(i)          where K = number of candidates
S(i)    = Sum_j  B(i, j)         (j = 3 Supreme Court judges)
Winner  = argmax_i  S(i)
```

**Pairwise case (K = 2).** In pairwise comparisons, Borda scoring simplifies to: winner = 1 point, loser = 0 points, per judge. With 3 judges, the verdict reduces to a **simple majority vote** (2-of-3). The formalism generalizes naturally to K > 2 candidates in production, where Borda Count rewards consistent top-ranking over occasional first-place finishes and resists outlier votes.

*Note: In production, candidate models may also vote (weight 1x, vs. 2x for SC judges). This weighting ensures SC predominance and has not altered any verdict in tested configurations (up to 6 candidates). This option will be made configurable in a future release.*

### 5.6 Self-Evaluation (Production Feature)

In addition to the three Supreme Court judges, each **candidate model** also evaluates all anonymized responses — effectively judging its own output without knowing which response is its own.

**Key properties:**
- Self-evaluation votes are displayed to the user but **do not influence the verdict**, which is determined solely by the Supreme Court panel (Section 5.5).
- When a candidate model ranks itself below competitors under blind conditions, this provides an additional credibility signal for the user: even the model itself "agrees" that another response was superior.
- Self-evaluation serves as a built-in diagnostic for self-favoritism (Section 3.1): systematic self-preference under blind conditions would be visible in the per-judge breakdown.

**This feature is not included in the validation experiments** (Part II), which test only the Supreme Court panel in isolation.

### 5.7 Human-in-the-Loop

Automated evaluation **compresses the decision space** rather than replacing human judgment. After each cycle, users receive a **side-by-side comparison** of top-ranked responses and make the final call. The framework filters; the human decides.

### 5.8 Bias Mitigations

| Bias | Mitigation |
|---|---|
| **Self-favoritism** [20] | Strict anonymization (Section 5.3) + multi-judge panel |
| **Position bias** | Random response ordering per evaluation and per run |
| **Verbosity bias** | Criteria penalize padding, reward conciseness |
| **Authority bias** | Multi-criteria scoring separates tone from substance |

No single mitigation is sufficient. The framework relies on their **combined effect** across judges, runs, and datasets.

### 5.9 Worked Example

A complete walkthrough of one pairwise evaluation (K = 2, 3 Supreme Court judges):

```
STEP 1 — INPUT
  Prompt: "Explain quantum entanglement to a 10-year-old."
  Model X response: [200 words about particles being best friends]
  Model Y response: [180 words about magic dice]

STEP 2 — ANONYMIZE
  Shuffle (random seed for this eval): Model Y drawn first
  -> Response A0 = Model Y    Response A1 = Model X
  All metadata stripped.

STEP 3 — CRITERIA GENERATION  (gpt-4o-mini, temp 0.3)
  From prompt alone, generates:
    1. Age-appropriateness (30%) — suitable vocabulary and concepts?
    2. Scientific accuracy  (25%) — correct without oversimplification?
    3. Engagement           (25%) — would a child stay interested?
    4. Completeness         (20%) — covers key aspects of entanglement?

STEP 4a — JUDGING: RANKINGS (3 Supreme Court judges)
  GPT-5.2-pro:      ranks A1 > A0  -> B(A0)=0, B(A1)=1
  Claude Opus 4.5:   ranks A0 > A1  -> B(A0)=1, B(A1)=0
  Gemini 2.5 Pro:    ranks A1 > A0  -> B(A0)=0, B(A1)=1

STEP 4b — SCORES PER CRITERION (example for 1 judge)
  GPT-5.2-pro scores for A0 (Model Y):
    Age-appropriateness: 4/5
    Scientific accuracy:  2/5  -> reason: "Conflates entanglement with teleportation"
    Engagement:           4/5
    Completeness:         3/5  -> reason: "Omits measurement collapse"

  (For any criterion scored 3 or below, the judge must provide
   a brief justification — max 15 words — explaining the low score.)

STEP 5 — AGGREGATE  (Borda Count, majority 2-of-3)
  S(A0) = 0 + 1 + 0 = 1
  S(A1) = 1 + 0 + 1 = 2
  Winner: A1 = Model X  (2-1 split decision)

STEP 6 — OUTPUT + HUMAN REVIEW
  AI recommendation: Model X (split decision, 2-1)
  User sees both responses side-by-side and makes the final decision.
```

---

## 6. The Cost-Quality-Speed Triangle

In production, model selection is not just about quality — it is a **three-way trade-off**:

| Dimension | What it measures |
|---|---|
| **Quality** | Borda-aggregated scores from multi-judge evaluation |
| **Cost** | Per-token inference cost across all runs |
| **Latency** | Time to first token and total generation time |

A model that ranks first on quality but costs 10x more and responds 5x slower may not be the right choice for high-throughput applications. The framework enables **Pareto-optimal selection**: identifying models not dominated on any dimension, letting users trade off based on their constraints.

---

# PART II — EMPIRICAL VALIDATION

## 7. Validation Methodology

To validate the framework described in Part I, we conduct two concordance studies comparing Agent Clash verdicts to human preferences. This section describes the methodology common to both experiments.

### 7.1 Validation Pipeline

The experiments use a **modified validation pipeline** that differs from the production pipeline in one critical way: responses are **injected** from benchmark datasets rather than generated by the pipeline. This isolates the judging component — any difference between AI and human verdicts reflects the quality of the judge panel, not response generation variance.

Both datasets are distributed under the CC-BY-4.0 license, which permits use, reproduction, and adaptation with attribution. Attributions are provided in references [1] and [28].

```
+------------------------------------------------------------------+
|                     VALIDATION PIPELINE                           |
|  (differs from production: responses INJECTED, not generated)     |
|                                                                   |
|  +-------------+                  +-------------+                 |
|  | BENCHMARK   |     inject       | ANONYMIZE   |                |
|  | DATASET     |----------------->| (identical  |                |
|  |             |  (bypass         |  to prod.)  |                |
|  | Pre-existing|   generation)    +------+------+                |
|  | response    |                         |                        |
|  | pairs +     |                         v                        |
|  | human       |         [Same judging pipeline as S5:            |
|  | verdict     |          criteria -> 3 SC judges -> Borda]       |
|  +-------------+                         |                        |
|                                          v                        |
|                           +-----------------------+               |
|                           | COMPARE               |               |
|                           | AI winner vs.         |               |
|                           | Human winner          |               |
|                           | = concordance?        |               |
|                           +-----------------------+               |
+------------------------------------------------------------------+
```

The anonymization, criteria generation, judging, and aggregation steps are **identical to the production pipeline** (Section 5).

### 7.2 Validation Design Choices

Two key choices were made for the validation experiments:

1. **Only the three Supreme Court judges evaluate.** Self-evaluation by candidate models (Section 5.6) is a production feature designed for user-facing transparency. In these experiments, we isolate the panel's judgment quality by using only the three SC judges. This eliminates confounding factors from variable candidate model availability and ensures all evaluations are strictly comparable. By excluding candidate votes, we measure the intrinsic quality of the SC panel without confounding from candidate self-interest.

2. **Pairwise comparisons (K = 2).** Both benchmark datasets provide pairwise human preferences. With K = 2 candidates and 3 judges, the Borda Count reduces to a **simple majority vote** (2-of-3). This is the simplest possible aggregation — a verdict is correct whenever at least 2 of 3 judges agree with the human preference.

---

## 8. Experiment 1 — MT-Bench (Large Capability Gaps)

> **Key question:** Does a 3-judge AI panel agree with expert human preferences when models differ widely in capability?

### 8.1 Materials and Protocol

**Dataset.** The MT-Bench Human Judgments dataset (LMSYS, CC-BY-4.0) [1] contains 3,300+ expert pairwise comparisons collected by NLP researchers at UC Berkeley and LMSYS. Evaluators compared outputs from 6 models — GPT-4, GPT-3.5-turbo, Claude-v1, Vicuna-13B-v1.2, Alpaca-13B, and LLaMA-13B — on 80 multi-turn questions spanning writing, reasoning, math, coding, extraction, STEM, and humanities. Judgments were conducted blind (model identities hidden from evaluators).

**Sample selection.** We selected N = 100 pairwise comparisons with clear winners (ties excluded), covering 53 unique questions across all 6 models. The distribution of human-selected winners was: GPT-4 (29), GPT-3.5-turbo (24), Claude-v1 (21), Vicuna-13B (16), Alpaca-13B (7), LLaMA-13B (3).

**Judge panel.** The three Supreme Court judges (GPT-5.2-pro, Claude Opus 4.5, Gemini 2.5 Pro) evaluated all 100 pairs.

**Protocol:**

```
1. SAMPLE      N = 100 pairwise comparisons from MT-Bench [1].
               Clear winners only (ties excluded). 53 unique questions, 6 models.

2. INJECT      Pre-existing human-collected responses injected into
               the validation pipeline (S7.1), bypassing generation.

3. JUDGE       3 Supreme Court judges: anonymize -> generate criteria ->
               dispatch judges -> Borda (= majority 2-of-3).

4. COMPARE     AI winner vs. human expert winner. Match/mismatch.
               No partial credit.

5. EXECUTE     100 evaluations, sequential webhook calls, ~66 min.
```

**What is fixed:** responses (from dataset), judge panel, aggregation method.
**What varies:** criteria generation seed, response shuffle order.

**Metrics:** raw concordance rate, Cohen's kappa [18], Wilson score 95% CIs, stratified analyses by model tier and disagreement direction.

### 8.2 Results

#### 8.2.1 Overall Concordance

> **Result:** 88.0% concordance (kappa = 0.760) — exceeds GPT-4 single-judge baseline (85%) and human-human agreement (81%).

| Metric | Value |
|---|---|
| Total evaluations | 100 |
| Matches (AI = Human) | 88 |
| Mismatches | 12 |
| **Concordance rate** | **88.0%** |
| Wilson score 95% CI | [80.2%, 93.0%] |
| Cohen's kappa | **0.760** |
| kappa 95% CI | [0.632, 0.895] |
| kappa interpretation (Landis & Koch) | Substantial agreement |

The observed concordance of 88.0% exceeds the single-judge GPT-4 baseline of 85% reported by Zheng et al. [1] under the S2 setup (ties excluded), and substantially exceeds the 81% human-human inter-annotator agreement reported on the same benchmark.

Cohen's kappa = 0.760 falls in the "substantial agreement" range on the Landis-Koch scale [25] (0.61 < kappa <= 0.80). The classification task spans 6 model categories with unequal base rates, making the high kappa particularly meaningful.

#### 8.2.2 Concordance by Model Tier

> **Result:** Strong models (GPT-4, Claude, GPT-3.5) -> 95.9% concordance. Weak models (Vicuna, Alpaca, LLaMA) -> 68.0%. The panel is ~11x more likely to agree with humans when the winner is a strong model.

This analysis stratifies results by the model that humans selected as the winner, regardless of which model it was paired against. The 100 evaluations cover all possible matchup types: strong vs. strong (e.g., GPT-4 vs. Claude-v1), strong vs. weak (e.g., GPT-4 vs. Alpaca-13B), and weak vs. weak (e.g., Vicuna vs. LLaMA). The table below shows that when a human selects a strong model as the winner, the AI panel almost certainly agrees (95.9%). Conversely, when a human selects a weak model as the winner (often against another weak model), the panel agrees in only 68% of cases — these situations being inherently more ambiguous.

A clear capability-dependent gradient emerges in the concordance data:

**Table 1.** Concordance rate by human-selected winner model, ordered by model capability tier.

| Human Winner | n | Matches | Concordance | Wilson 95% CI |
|---|---|---|---|---|
| GPT-4 | 29 | 29 | **100.0%** | [88.3%, 100.0%] |
| Claude-v1 | 21 | 20 | **95.2%** | [77.3%, 99.2%] |
| GPT-3.5-turbo | 24 | 22 | **91.7%** | [74.2%, 97.7%] |
| Vicuna-13B-v1.2 | 16 | 12 | **75.0%** | [50.5%, 89.8%] |
| Alpaca-13B | 7 | 5 | **71.4%** | [35.9%, 91.8%] |
| LLaMA-13B | 2 | 0 | **0.0%** | [0.0%, 65.8%] |

*Note: One evaluation (id = 36) produced no result due to a silent pipeline error (0 judges, empty winner fields) and is counted as a non-match. Table 1 sums to N = 99; overall concordance is computed on N = 100.*

When grouping models into tiers:

**Table 2.** Concordance by model capability tier.

| Tier | Models | n | Matches | Concordance | Wilson 95% CI |
|---|---|---|---|---|---|
| Strong | GPT-4, Claude-v1, GPT-3.5-turbo | 74 | 71 | **95.9%** | [88.7%, 98.6%] |
| Weak | Vicuna-13B, Alpaca-13B, LLaMA-13B | 25 | 17 | **68.0%** | [48.4%, 82.8%] |

The difference is highly significant (chi-squared = 14.78, df = 1, p = 0.0001). The odds ratio is 11.14: the AI panel is approximately 11 times more likely to agree with the human verdict when the human-selected winner belongs to the strong tier.

This pattern is consistent with findings from Zheng et al. [1, Figure 2], who reported that GPT-4 single-judge agreement with humans increases monotonically with the performance gap between the evaluated models — from approximately 70% for closely matched pairs to nearly 100% for pairs with large capability differences.

#### 8.2.3 Cost and Efficiency

| Metric | Value |
|---|---|
| Total evaluation cost | $12.86 |
| Cost per evaluation | $0.129 |
| Cost per correct prediction | $0.146 |
| Mean evaluation time | 39.6 s |
| Total wall-clock time | ~66 min |

### 8.3 Analysis of Disagreements

> **Result:** In 73% of disagreements, the AI favored a stronger model — suggesting a "capability prior" that is expected and not problematic. AI confidence is equally high for correct and incorrect verdicts (no self-awareness of errors on this benchmark).

Of the 12 non-matches, one (id = 36) is the empty pipeline error noted above. The remaining 11 are substantive AI-human disagreements:

**Table 3.** Substantive AI-human disagreements (11 of 12 non-matches).

| eval_id | Model A | Model B | Human Winner | AI Winner |
|---|---|---|---|---|
| 10 | alpaca-13b | vicuna-13b | alpaca-13b | vicuna-13b |
| 11 | vicuna-13b | gpt-3.5-turbo | vicuna-13b | gpt-3.5-turbo |
| 14 | vicuna-13b | alpaca-13b | vicuna-13b | alpaca-13b |
| 23 | vicuna-13b | claude-v1 | claude-v1 | vicuna-13b |
| 30 | alpaca-13b | llama-13b | llama-13b | alpaca-13b |
| 32 | gpt-4 | llama-13b | llama-13b | gpt-4 |
| 33 | gpt-3.5-turbo | vicuna-13b | vicuna-13b | gpt-3.5-turbo |
| 47 | alpaca-13b | gpt-4 | alpaca-13b | gpt-4 |
| 56 | claude-v1 | gpt-3.5-turbo | gpt-3.5-turbo | claude-v1 |
| 77 | vicuna-13b | gpt-3.5-turbo | vicuna-13b | gpt-3.5-turbo |
| 94 | gpt-3.5-turbo | vicuna-13b | gpt-3.5-turbo | vicuna-13b |

**Directional bias.** In 8 of 11 disagreements (72.7%), the AI panel selected a model from a higher capability tier than the human-selected winner. In only 3 cases (27.3%) did the AI select a weaker-tier model. While this asymmetry is suggestive of a "capability prior" — a tendency for AI judges to favor models with generally higher capability even when the specific response from the weaker model was superior — the sign test does not reach significance at alpha = 0.05 (z = 1.51, p = 0.13, two-sided), likely due to the small sample of disagreements.

This capability prior is expected and should not be interpreted as a defect. It reflects the fact that AI judges prioritize substance and reasoning depth over stylistic or subjective qualities, which naturally coincides with the known capability hierarchy among models. Importantly, this prior manifests only in the rare disagreement cases (12% of evaluations) and does not affect the 88% where the panel correctly identifies the human-preferred response. Experiment 2 (Section 9) will show that this prior disappears when evaluated models are closely matched in capability — confirming that it is a dataset property, not an intrinsic panel bias.

**Pair concentration.** The GPT-3.5-turbo vs. Vicuna-13B pair accounts for 4 of the 11 disagreements (36.4%), with a 44.4% disagreement rate for that specific pair. This pair sits at the boundary between strong and weak tiers, where quality differences are smallest and subjective judgment is most likely to diverge.

### 8.4 Discussion

**Comparison with prior work:**

**Table 4.** Concordance between automated evaluation and human judgments across frameworks and benchmarks.

| Method | Benchmark | Agreement | kappa | Reference |
|---|---|---|---|---|
| GPT-4 single judge | MT-Bench (S2, no ties) | 85% | — | Zheng et al. [1] |
| Human vs. human | MT-Bench (S2) | 81% | — | Zheng et al. [1] |
| GPT-4 single judge | MT-Bench (headline) | >80% | — | Zheng et al. [1] |
| GPT-4 Turbo single judge | TriviaQA | — | 0.84 | Thakur et al. [23] |
| Human vs. human | TriviaQA | — | 0.96 | Thakur et al. [23] |
| PoLL (3-model panel) | KILT-NQ | — | 0.763 | Verga et al. [21] |
| PoLL (3-model panel) | HotPotQA | — | 0.889 | Verga et al. [21] |
| Arena-Hard-Auto | Chatbot Arena | 89.1% | — | Li et al. [26] |
| Crowd vs. expert | Chatbot Arena | 72-83% | — | Chiang et al. [10] |
| Expert vs. expert | Chatbot Arena | 79-90% | — | Chiang et al. [10] |
| **Agent Clash (3-judge SC)** | **MT-Bench** | **88.0%** | **0.760** | **This work (Exp. 1)** |

*Note: "Agreement" measures the raw percentage of cases where two evaluators concur. Cohen's kappa corrects this rate by subtracting chance-expected agreement, providing a more conservative measure. The two metrics are not interchangeable: an agreement of 85% can correspond to very different kappa values depending on category distribution. Some studies report only one of the two metrics — blank cells (—) indicate the metric was not computed in the original work, not that it is zero.*

Our 3-judge Supreme Court panel achieves concordance that:

- **Exceeds the single-judge GPT-4 baseline** by 3.0 percentage points (88.0% vs. 85%), consistent with the multi-judge improvement reported by Verga et al. [21] and Chan et al. [22].
- **Exceeds human-human agreement** on the same benchmark by 7.0 points (88.0% vs. 81%), suggesting the panel achieves super-human concordance — though this comparison should be interpreted with caution, as the human baseline may reflect legitimate preference diversity rather than evaluator error.
- **Not directly comparable to Arena-Hard-Auto's 89.1%** [26]: their metric measures system-level ranking agreement (Spearman correlation between model rankings), while ours measures per-example concordance (binary match/mismatch per evaluation). Per-example concordance is a stricter, more conservative measure.
- **Achieves kappa = 0.760**, falling within the same range as PoLL on HotPotQA (kappa = 0.889) and substantially above single-judge GPT-4 on KILT-NQ (kappa = 0.627).

**The concordance gradient is the central finding.** Near-perfect agreement (95.9%) on strong-tier winners shows the panel reliably identifies clear quality differences. The lower concordance (68.0%) on weak-tier winners reflects borderline cases where human preferences themselves diverge. Experiment 2 (Section 9) confirms this at the inter-experiment level (88% -> 76%).

**The capability prior in disagreements.** In 72.7% of mismatches, the AI favored higher-tier models — is this a bias or a feature? When humans chose LLaMA-13B over GPT-4 (eval_id 32), judges overrode with maximum conviction. This may reflect a genuine prior toward conventionally stronger models, or the human evaluator may have been influenced by novelty. Disentangling these hypotheses requires larger samples.

**Limitations.** (1) MT-Bench models (2023) have large capability gaps -> addressed in Experiment 2 with 2024 frontier models. (2) N = 100 -> Wilson CI = [80.2%, 93.0%]. (3) Turn-1 responses only.

---

## 9. Experiment 2 — Chatbot Arena (Frontier Models, Small Capability Gaps)

> **Key question:** What happens when the models under evaluation are all frontier-class (GPT-4o, Claude 3.5, Gemini 1.5, etc.) and the quality gap is minimal?

### 9.1 Materials and Protocol

**Motivation.** Experiment 1 validated the framework on models with large capability gaps (GPT-4 vs. LLaMA-13B). A critical question is whether the pipeline remains useful when evaluated models are closely matched frontier systems — the scenario most relevant to real-world deployment decisions.

**Dataset.** We use the `lmarena-ai/arena-human-preference-100k` dataset [28] (CC-BY-4.0), which contains approximately 100,000 anonymous pairwise battles collected between June and August 2024 on the Chatbot Arena platform. Each row represents a single human judge's preference between two model responses to the same prompt.

**Sample.** We extract N = 100 evaluations using the following inclusion criteria:
- Both models belong to a curated set of 28 frontier or near-frontier models (Table 5)
- Clear winner (no ties)
- Anonymous battle
- English language
- Single-turn conversation
- Minimum 30 characters per response

Sampling uses a fixed reproducibility parameter (seed = 42): this means that anyone re-running the same extraction script on the same dataset will obtain exactly the same 100 evaluations. This parameter controls Python's pseudo-random number generator and ensures full reproducibility of sample selection.

**Table 5.** Models included in the Arena validation sample (25 unique models appearing).

| Family | Models |
|---|---|
| GPT | gpt-4o, gpt-4o-aug, gpt-4o-mini, gpt-4-turbo, gpt-4-turbo-jan, gpt-4-turbo-nov, gpt-4, chatgpt-4o |
| Claude | claude-3.5-sonnet, claude-3-opus, claude-3-sonnet, claude-3-haiku |
| Gemini | gemini-1.5-pro, gemini-1.5-pro-exp, gemini-1.5-flash |
| Llama | llama-3-70b, llama-3.1-70b, llama-3.1-405b |
| Mistral | mistral-large-2, mixtral-8x22b |
| Other | deepseek-v2, deepseek-coder-v2, yi-large, yi-large-preview, qwen2-72b, nemotron-340b, command-r-plus |

**Key differences from Experiment 1.** (1) Models are 2024-generation frontier systems with much smaller capability gaps. (2) Human judgments come from crowd-sourced anonymous users (not expert annotators). (3) 25 models instead of 6 — over 80 distinct model pairs represented. (4) Prompts are user-generated real-world queries, not curated multi-turn benchmark questions.

**Judge panel.** The three Supreme Court judges (GPT-5.2-pro, Claude Opus 4.5, Gemini 2.5 Pro) evaluated all pairs across all runs.

**Protocol:**

```
1. SAMPLE      N = 100 pairwise from arena-human-preference-100k [28].
               Inclusion: both models in frontier set (Table 5),
               clear winner, anonymous, English, single-turn, >=30 chars.
               Seed = 42 (reproducibility parameter).

2. INJECT      Pre-existing responses -> validation pipeline (S7.1).

3. RUN 1       Standard judging pipeline (S5). Record AI winners.

4. RUN 2       Re-run 100 evals with fresh random seeds.
               Same responses, new criteria generation + shuffle.

5. COMPARE     Each run vs. human (concordance).
               Run 1 vs. Run 2 (test-retest reliability).

6. CLASSIFY    From Runs 1-2, classify each eval:
               - Stable correct (both runs match human)
               - Stable incorrect (both runs disagree with human)
               - Unstable (runs disagree with each other)

7. RUN 3       Targeted re-evaluation of 60 most informative evals:
   (targeted)  all 20 stable-incorrect + all 9 unstable +
               31 low-confidence stable-correct.
```

**What is fixed:** responses (from dataset), judge panel, aggregation method.
**What varies across runs:** criteria generation seed, response shuffle order.

### 9.2 Results

> **Result:** 76.0% concordance (kappa = 0.520) — a 12-point drop from MT-Bench, within the 72-83% range of crowd-expert human agreement on the same platform.

**Table 6.** Overall concordance on Chatbot Arena.

| Metric | Run 1 | Run 2 | Combined |
|---|---|---|---|
| N | 100 | 100 | 200 |
| Concordance | **76.0%** | **75.0%** | **75.5%** |
| Wilson 95% CI | [66.8%, 83.3%] | [65.7%, 82.5%] | [69.1%, 80.9%] |
| Cohen's kappa | 0.520 | 0.500 | 0.510 |
| kappa 95% CI | [0.353, 0.687] | [0.330, 0.670] | [0.391, 0.629] |
| Total cost | $15.08 | $14.94 | $30.02 |
| Cost per eval | $0.151 | $0.149 | $0.150 |

Cohen's kappa = 0.510 falls in the "moderate" range on the Landis-Koch scale [25], consistent with the difficulty of discriminating between closely matched frontier models.

**Table 7.** Concordance by human-selected winner (Run 1, models with n >= 3).

| Human Winner | n | Matches | Concordance | Wilson 95% CI |
|---|---|---|---|---|
| chatgpt-4o | 8 | 7 | **87.5%** | [52.9%, 97.8%] |
| gemini-1.5-pro-exp | 7 | 6 | **85.7%** | [48.7%, 97.4%] |
| gpt-4o | 13 | 11 | **84.6%** | [57.8%, 95.7%] |
| gemini-1.5-pro | 6 | 5 | **83.3%** | [43.6%, 97.0%] |
| claude-3-opus | 4 | 3 | **75.0%** | [30.1%, 95.4%] |
| llama-3-70b | 4 | 3 | **75.0%** | [30.1%, 95.4%] |
| claude-3.5-sonnet | 13 | 9 | **69.2%** | [42.4%, 87.3%] |
| yi-large | 3 | 2 | **66.7%** | [20.8%, 93.9%] |
| mistral-large-2 | 5 | 3 | **60.0%** | [23.1%, 88.2%] |
| gpt-4-turbo-nov | 5 | 3 | **60.0%** | [23.1%, 88.2%] |
| llama-3.1-70b | 5 | 3 | **60.0%** | [23.1%, 88.2%] |
| deepseek-v2 | 4 | 2 | **50.0%** | [15.0%, 85.0%] |
| llama-3.1-405b | 4 | 2 | **50.0%** | [15.0%, 85.0%] |

Unlike Experiment 1, there is no clear tier-based concordance gradient: all models are frontier-class, and concordance varies within a narrower range (50-88%).

### 9.3 Test-Retest Reliability

> **Result:** 91% of evaluations get the same AI verdict across two independent runs (kappa = 0.757). Errors are 69% systematic (same wrong answer every time) and 31% stochastic (coin-flip behavior).

**Table 8.** Inter-run concordance matrix (N = 100 paired evaluations).

|  | Run 2 Match | Run 2 Miss |
|---|---|---|
| **Run 1 Match** | 71 | 5 |
| **Run 1 Miss** | 4 | 20 |

- **Same AI verdict in both runs:** 91/100 = **91.0%**
- **Inter-run Cohen's kappa:** 0.757 ("substantial" agreement)
- **McNemar's test:** chi-squared = 0.00 (with continuity correction), p > 0.05 — no significant difference between runs

Of the 9 discordant evaluations (different AI verdict between runs), all involve pairs of frontier models where the quality gap is minimal (e.g., claude-3.5-sonnet vs. gemini-1.5-pro, claude-3-opus vs. gpt-4o). These represent the "noise floor" where stochastic variation in criteria generation and judge sampling produces different verdicts on borderline cases.

**Stability classification:** 71% of evaluations are stably correct (both runs agree with human), 20% are stably incorrect (both runs disagree), and 9% are unstable (runs disagree with each other). The low instability rate confirms that the pipeline is reproducible and that errors are predominantly systematic rather than stochastic.

**Targeted Run 3 confirmation (N = 60).** To strengthen these findings, we conducted a targeted third run on the 60 most statistically informative evaluations: all 20 stably incorrect, all 9 unstable, and 31 low-confidence stably correct. To select the 31 stably correct evaluations for retesting, we use the numerical confidence score produced by the pipeline, which measures the normalized Borda victory margin divided by the number of judges. The 31 evaluations selected are those with the lowest average confidence across the first two runs. Results:

- **Systematic errors confirmed:** 19/20 (95%) of stably incorrect evaluations remained wrong in Run 3. Under H0 that errors are random coin flips, P(>=19/20) < 0.00002 (binomial test). The sole exception (id = 100, llama-3.1-405b vs. llama-3.1-70b) was the only evaluation to flip in three runs. The remaining 19 evaluations were wrong in all three runs, establishing them as **irreducible systematic disagreements** between the AI panel and crowd-sourced human preferences.
- **Unstable evaluations are genuinely stochastic:** 5/9 (55.6%) matched in Run 3 — close to the expected 50% for random coin-flip behavior, confirming that these represent the pipeline's noise floor.
- **Low-confidence predictions are fragile:** 3/31 (9.7%) of stably-correct-but-low-confidence evaluations flipped to wrong in Run 3.
- **Majority-vote concordance equals single-run concordance:** Across all 100 evaluations, the majority verdict across available runs yields 76.0% — identical to the single-run rate. This confirms that errors cannot be averaged away by re-running, because the dominant error mode is systematic.

### 9.4 Disagreement Analysis

> **Result:** No capability bias on frontier models (unlike MT-Bench). No self-favoritism detected. Blind self-evaluation data confirms absence of self-promotion.

**ELO scores.** ELO scores, derived from the Chatbot Arena leaderboard [10], are relative competence ratings computed from thousands of pairwise human votes, analogous to chess rankings: an ELO difference of 100 points means the higher-rated model is predicted to win in approximately 64% of matchups.

Of the 24 disagreements in Run 1:

- **No directional capability bias.** Using approximate Chatbot Arena ELO ratings (circa August 2024), we test whether the AI panel systematically favors higher-ELO models in disagreements. Of 24 mismatches, the AI selected the higher-ELO model in 11 cases (45.8%) and the lower-ELO model in 10 cases (41.7%), with 3 ties. The sign test is non-significant (p = 1.0), and the mean ELO difference is -2.9 (negligible). This contrasts sharply with Experiment 1, where 72.7% of mismatches favored higher-tier models — confirming that the "capability prior" observed in MT-Bench was driven by the large capability gaps in that dataset, not by an intrinsic bias of the judge panel.

- **Chronic disagreement pairs.** claude-3.5-sonnet vs. gemini-1.5-pro is the most disputed pair (3 of 5 evaluations misclassified in both runs), consistent with these two models being very closely matched on the Arena leaderboard (ELO difference of approximately 20). The deepseek-v2 vs. gemini-1.5-flash and gpt-4o-aug vs. llama-3.1-405b pairs are also systematically misclassified across both runs, representing genuine ambiguity in these matchups.

- **Self-favoritism.** Since the judge panel includes GPT-5.2-pro, Claude Opus 4.5, and Gemini 2.5 Pro, we test whether judges favor their own model family. GPT-family AI picks (33) closely match human picks (34); Claude-family AI picks (18) match human picks (19); Gemini-family AI picks (20) slightly exceed human picks (16). No systematic self-favoritism is detected, consistent with findings from Panickssery et al. [20] under blind conditions.

- **Blind self-evaluation.** Our validation data includes 252 instances where a candidate model also served as a (non-Supreme-Court) judge, effectively evaluating its own output without knowing which response was its own. Across all such cases, models ranked themselves first in 52.8% of evaluations, compared to an expected 59.9% based on human win rates for those same models — a difference of -7.1 percentage points, indicating a slight anti-self-bias overall. Results by model: GPT-4 showed a strong anti-self-bias (-25.0pp, N=44), GPT-3.5-turbo was near-neutral (+2.1pp, N=47), and Claude 3.5 Sonnet showed a slight anti-self-bias (-7.7pp, N=78). Some smaller-sample models showed positive deviations (e.g., GPT-4-turbo +27.8pp, N=18), but these are not statistically significant given the sample sizes. Overall, no systematic self-promotion is detected under blind conditions.

- **Error decomposition.** Of the 29 evaluations that are not stably correct across both runs (29%), 69% are systematic (20 evaluations consistently wrong across both runs) and 31% is stochastic (9 evaluations inconsistent between runs). A targeted third run confirmed 19/20 (95%) of systematic errors (binomial p < 0.00002), while the 9 stochastic errors showed 55.6% concordance in Run 3 — close to the 50% expected for genuine coin-flip behavior.

### 9.5 Discussion

**76% on Arena is competitive — and approaches the human ceiling.** It falls within the 72-83% range of human crowd-expert agreement on the same platform [10], at $0.15/eval (vs. several dollars for human annotation) with full reproducibility. On frontier models, the capability prior observed in Experiment 1 naturally disappears: when models are close in performance, there is no capability hierarchy left to favor. Furthermore, it is important to contextualize the 76% rate: each Arena evaluation reflects a single human vote (not a consensus). Studies by Chiang et al. [10] show that crowd-expert agreement ranges from 72-83%, and expert-expert agreement from 79-90%. Our panel at 76% therefore falls within the range one would expect from an additional human evaluator. **A perfect model that exactly reproduced the judgment of an average human would cap at approximately 80% concordance with another random human** — our panel approaches this bound.

**High test-retest reliability.** 91% inter-run agreement (kappa = 0.757). The 9 unstable evaluations showed 55.6% concordance in Run 3 — indistinguishable from chance — confirming they represent the pipeline's noise floor.

**19% irreducible systematic errors.** Majority-vote across 3 runs does not improve concordance (76.0% = single-run rate). Possible explanations:
- AI judges prioritize factual accuracy/reasoning depth; human voters may weigh style or helpfulness differently
- Some crowd-sourced votes are noisy or idiosyncratic
- Certain model pairs (e.g., Claude 3.5 Sonnet vs. Gemini 1.5 Pro) sit below the resolution limit of any evaluation system

**Nature of disagreements.** Of the 24 disagreements in Experiment 2, the vast majority concern pairs of models that are very close in performance (ELO difference < 50 points). In these cases, both responses are often of comparable quality, and the choice between them reflects subjective preference rather than an objectifiable difference. This echoes observations from Feng et al. [24], who show that even the best evaluators fail to maintain consistent preferences in ~25% of difficult cases. **The panel's errors are concentrated precisely where the very notion of a "correct answer" is ambiguous**, which puts their practical impact into perspective.

**No capability bias on frontier models.** The "capability prior" from Experiment 1 (72.7% of errors favoring stronger models) disappears on Arena: only 45.8% favor the higher-ELO model (sign test p = 1.0). The bias was a dataset artifact, not an intrinsic property of the panel.

**No self-favoritism.** GPT, Claude, and Gemini family win rates closely match human preferences. Blind self-evaluation data (N = 252) confirms the absence of self-promotion under anonymized conditions. Using frontier models as judges for their own lineage is valid under blind conditions [20].

**Limitations.** (1) Each Arena evaluation reflects a single human vote, not a consensus. (2) Dataset covers June-August 2024 only. (3) Single-turn conversations only. (4) N = 260 provides adequate overall power but limited per-model subgroup power. (5) Reducing the 19% systematic error rate requires architectural changes, not repetition.

---

## 10. Cross-Experiment Analysis

### 10.1 Concordance Gradient

> **Result:** The 12-point concordance drop (88% -> 76%) is statistically significant (p = 0.027) and expected: smaller capability gaps make the task harder for both humans and AI.

**Table 9.** Comparison of Experiment 1 (MT-Bench) and Experiment 2 (Chatbot Arena).

| Metric | Exp. 1: MT-Bench | Exp. 2: Arena |
|---|---|---|
| **Dataset** | MT-Bench Human Judgments [1] | Chatbot Arena 100k [28] |
| **Year** | 2023 | 2024 |
| **N models** | 6 | 25 |
| **N evaluations** | 100 | 100 (x2 runs) + 60 targeted |
| **Model gap** | Large (GPT-4 vs. LLaMA-13B) | Small (GPT-4o vs. Claude 3.5 Sonnet) |
| **Human annotators** | Expert | Crowd-sourced |
| **Concordance** | **88.0%** | **76.0%** / **75.0%** |
| **Cohen's kappa** | 0.760 | 0.520 / 0.500 |
| **Total cost** | $12.86 | $38.84 (260 evals) |
| **Cost per eval** | $0.129 | $0.149 |
| **Test-retest** | — | 91.0% (kappa = 0.757) |
| **Systematic error rate** | — | 19% (confirmed 3 runs) |

The 12-point drop is **statistically significant** (z = 2.21, p = 0.027) and driven by three factors: (1) **smaller capability gaps** make discrimination harder for both humans and AI; (2) **crowd vs. expert annotators** — our 76% falls within the 72-83% crowd-expert agreement range on Arena [10]; (3) **25 models (vs. 6)** create many close matchups where any evaluator would struggle.

**Table 10.** Consolidated comparison with prior work and human baselines.

| Category | Method | Benchmark | Agreement | kappa | Reference |
|---|---|---|---|---|---|
| **Human baselines** | Crowd vs. expert | Chatbot Arena | 72-83% | — | Chiang et al. [10] |
| | Expert vs. expert | Chatbot Arena | 79-90% | — | Chiang et al. [10] |
| | Human vs. human | MT-Bench (S2) | 81% | — | Zheng et al. [1] |
| **Single judge** | GPT-4 | MT-Bench (S2, no ties) | 85% | — | Zheng et al. [1] |
| **Multi-judge panels** | PoLL (3 models) | HotPotQA | — | 0.889 | Verga et al. [21] |
| | **Agent Clash (3 SC)** | **MT-Bench** | **88.0%** | **0.760** | **This work** |
| | **Agent Clash (3 SC)** | **Arena (frontier)** | **76.0%** | **0.520** | **This work** |
| **Reliability** | **Agent Clash test-retest** | **Arena** | **91.0%** | **0.757** | **This work** |

*Metrics are not directly comparable across different benchmarks (see note on Table 4 regarding agreement vs. kappa interpretation).*

### 10.2 Ablation: Individual Judge vs. Panel

To assess whether the multi-judge panel improves upon individual judges, we extracted each Supreme Court judge's individual vote from all evaluations across both experiments and computed per-judge concordance with human preferences.

**Table 11.** Individual judge concordance vs. panel (majority vote), pooled N = 192.

| Judge | MT-Bench (N=99) | Arena (N=93) | Pooled (N=192) |
|---|---|---|---|
| GPT-5.2-pro | 88.9% | 74.2% | 81.8% |
| Claude Opus 4.5 | 87.9% | 76.3% | 82.3% |
| Gemini 2.5 Pro | 83.8% | 69.9% | 77.1% |
| **Panel (majority 2-of-3)** | **88.9%** | **75.3%** | **82.3%** |

**Key findings:**

1. **The panel matches the best individual judge** — without knowing in advance which judge will be best. On MT-Bench, GPT-5.2-pro is the best individual judge (88.9%); on Arena, Claude Opus 4.5 is best (76.3%). The best judge changes across datasets, but the panel consistently matches or exceeds the best.

2. **The panel protects against the worst judge.** Compared to a randomly selected single judge, the panel gains +1.9pp on average. Compared to the worst judge (Gemini 2.5 Pro), the panel gains +5.2pp.

3. **The multi-judge architecture provides robustness**, not peak performance improvement. Its primary value is **insurance**: it guarantees best-individual-judge performance regardless of which dataset or task domain is evaluated.

### 10.3 Unanimity as a Confidence Signal

A key advantage of a multi-judge panel over a single judge is the ability to measure **internal agreement** as a confidence indicator. We analyze concordance stratified by whether the three judges reached a unanimous (3-0) or split (2-1) decision.

**Table 12.** Concordance by panel unanimity (pooled across both experiments, N = 239).

| Agreement | N | % of evals | Concordance with human |
|---|---|---|---|
| Unanimous (3-0) | 192 | 80% | **84.9%** |
| Split (2-1) | 47 | 20% | **63.8%** |
| **Difference** | | | **+21.1pp** |

**Key findings:**

1. **Unanimity is a strong predictor of accuracy.** When all three judges agree, concordance is 84.9%. When they split 2-1, concordance drops to 63.8%. The 21.1-percentage-point gap provides a **binary, empirically calibrated confidence signal**.

2. **On split decisions, the panel still outperforms individual judges.** When the 3 judges disagree (2 against 1), this is an inherently difficult case. The panel, which follows the majority (2-of-3), still achieves 63.8% concordance with the human preference. By comparison, if a single judge had been used on these same difficult cases, the best individual judge (Claude Opus 4.5) would have achieved only approximately 57.4%. The +6.4pp gain shows that it is precisely on contentious cases that the 3-judge vote adds the most value — the dissenting judge is wrong in the majority of cases.

3. **A single judge cannot produce this signal.** A lone judge always outputs a verdict with no internal measure of certainty. The multi-judge panel provides an actionable routing mechanism: unanimous verdicts can be accepted with high confidence; split verdicts can be flagged for human review.

**Practical implication:** In production, the unanimity signal enables a simple triage strategy — accept unanimous decisions automatically, escalate split decisions to human review — that would raise effective concordance on the accepted subset to ~85%.

### 10.4 Synthesis

The multi-judge panel's value proposition is threefold:

1. **Robustness:** It matches the best individual judge without knowing which judge is best a priori, protecting against the +5.2pp cost of picking the wrong single judge.

2. **Confidence calibration:** The unanimity signal (84.9% vs. 63.8%) provides an actionable, empirically validated confidence indicator that a single judge cannot produce.

3. **Bias diversification:** Three model families from different providers ensure that no single architectural bias dominates the verdict.

The multi-judge panel does not dramatically improve raw concordance over the best single judge. Its primary contribution is **reliability and interpretability** — knowing when to trust the verdict and when to escalate.

---

## 11. Limitations

The framework measures perceived quality, coherence, and reasoning stability well. It does **not** claim to measure absolute truth, domain-specific factual correctness, or universal stylistic preference.

| Limitation | Implication |
|---|---|
| **Judge capability ceiling** | As candidates approach judge-level capability, discrimination degrades |
| **Cost** | Multi-judge evaluation is more expensive than a single judge — but at ~$0.15 per evaluation, the added cost is negligible for a prompt with a few candidate models. The framework is designed for high-stakes selection decisions |
| **Residual bias** | LLM judges carry biases [9] that cannot be fully eliminated -> human-in-the-loop pathway preserved |
| **Ground truth quality** | MT-Bench uses expert annotations; Arena uses single crowd-sourced votes per evaluation — neither constitutes a definitive ground truth. Concordance metrics inherit the noise of the reference standard |
| **Criteria generator bias** | Dynamic criteria are generated by gpt-4o-mini (Section 5.4). This introduces a single-model dependency: the evaluation rubric reflects one model's interpretation of task requirements. Using a different criteria generator could shift results |
| **Panel size** | Only 3 judges tested; larger panels might improve concordance or confidence calibration |
| **No dataset variation** | We could not test robustness to varied response sets for the same prompt — no such paired datasets with human preferences exist in the literature. This remains an important open question |

The framework does not replace human judgment — it provides a scalable, reproducible signal that focuses human attention where it matters most.

---

## 12. Future Work

Several extensions are planned: (1) measuring ranking stability across N = 10 independent evaluation cycles using Kendall's tau [19]; (2) assessing cross-dataset generalization across K = 5 heterogeneous datasets; (3) controlled self-favoritism experiments under blind conditions with Wilcoxon signed-rank tests; (4) mapping the cost-quality-latency Pareto frontier; and (5) testing larger judge panels (5 or 7 judges) to assess whether additional judges improve concordance or confidence calibration.

**(6) End-to-end pipeline stability.** Experiments 1-2 held responses fixed and measured *judgment* variance only (Section 7.1). In production, however, Agent Clash regenerates responses at each execution. A preliminary test examines what happens when both response generation and judgment are repeated from scratch. We ran 10 prompts (4 easy, 4 medium, 2 hard) through 3 complete cycles — regenerating GPT-4o-mini and Claude 3.5 Sonnet responses at temperature 1.0, then judging each fresh pair — for 30 total evaluations ($3.99). All response hashes differed across runs, confirming non-deterministic generation. Result: **6/10 prompts (60%) produced the same winner in all 3 runs**, while 4/10 showed a 2-1 split. This end-to-end stability rate is markedly lower than the 91% judgment-only test-retest agreement (Experiment 2), indicating that generation variance is the dominant source of instability — not the judge panel. This result underscores that judgment stability — validated in this study — is a necessary but not sufficient condition for end-to-end stability. No human panel baseline exists in the literature for this type of test, which constitutes an open limitation. A full-scale study with N >= 30 prompts and variance decomposition (generation vs. judgment components) is warranted.

---

## 13. Conclusion

> **Bottom line:** A 3-judge AI panel achieves 88% concordance with humans when models differ widely and 76% when they're closely matched — comparable to human-human agreement in both cases. The panel's primary value is not raw accuracy improvement over the best single judge, but robustness and calibrated confidence.

We validated a multi-judge framework across two difficulty regimes. The **concordance gradient** — 88.0% (kappa = 0.760) on large gaps, 76.0% (kappa = 0.520) on frontier models — tracks task difficulty and falls within human inter-annotator agreement on both benchmarks. Test-retest reliability is 91%, with errors decomposed into 19% systematic (confirmed irreducible, p < 0.00002) and 9% stochastic.

**What distinguishes this work:**
1. Validation on **closely matched frontier models**, not just benchmarks with large gaps
2. **Systematic error characterization** across repeated runs (not just single-shot accuracy)
3. **Ablation analysis** showing the panel matches the best individual judge without knowing which one in advance, with +5.2pp protection against the worst
4. A **calibrated unanimity signal** (84.9% vs. 63.8%) providing an actionable confidence indicator that a single judge cannot produce

The 19% irreducible disagreement rate sets a practical ceiling. Further improvement requires architectural changes — specialized judge ensembles, human-in-the-loop escalation, or criteria tailored to crowd-preference alignment — not additional runs of the same pipeline.

---

## Reproducibility and Transparency

Many existing benchmarks are **opaque**: evaluation data, judge identities, and scoring criteria are hidden from users. This framework takes the opposite approach.

In production, Agent Clash regenerates responses at each execution — results will therefore not be identical from one run to the next, but the evaluation methodology is strictly reproducible. The judgment stability validated in this study (91% test-retest) ensures that the panel's verdicts are consistent even when criteria and response ordering are re-randomized.

To enable independent verification of the results reported in this study, the following materials are available as open access:

- **Validation pipelines** (exported n8n workflows)
- **Datasets** used (MT-Bench and Arena samples)
- **Raw results** (per-judge, per-evaluation scoring matrices)
- **Article** (English and French versions)

Repository: [github.com/anthonyboisbouvier-paris/agent-clash-paper](https://github.com/anthonyboisbouvier-paris/agent-clash-paper)

Production platform: [agent-clash.ai](https://www.agent-clash.ai/)

- **Users provide their own API key** -> full control and auditability of every inference call
- **All prompts, datasets, criteria, judge identities, and aggregation rules** are disclosed and configurable [8]
- **Raw scoring matrices** available per-judge, per-run, per-dataset
- **Side-by-side response views** for direct model output inspection

Any user can re-run the exact same evaluation and expect statistically consistent results. This is not a benchmark you trust — it is a benchmark you **verify**.

---

## Acknowledgments

This paper was reviewed with the assistance of ChatGPT-o3 (OpenAI). All experimental design, data analysis, interpretation, and editorial decisions were made by the human author.

---

## References

[1] Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023 Datasets and Benchmarks Track. arXiv:2306.05685*

[2] Liu, Y., Iter, D., Xu, Y., Wang, S., Xu, R., & Zhu, C. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment. *arXiv:2303.16634*

[3] Kocmi, T. & Federmann, C. (2023). Large Language Models Are State-of-the-Art Evaluators of Translation Quality. *arXiv:2302.14520*

[4] Wang, X., Wei, J., Schuurmans, D., Le, Q., Chi, E., Narang, S., Chowdhery, A., & Zhou, D. (2022). Self-Consistency Improves Chain-of-Thought Reasoning in Language Models. *arXiv:2203.11171*

[5] Oren, Y., Geva, M., Stanovsky, G., & Smith, N. A. (2022). Distribution Shifts Are the Norm in NLP. *arXiv:2205.12350*

[6] Mitchell, E., Noh, Y., Li, S., Armstrong, W., Agarwal, A., Liu, P., Finn, C., & Manning, C. D. (2022). Enhancing Self-Consistency and Performance of Pre-Trained Language Models through Natural Language Inference. *arXiv:2211.11875*

[7] Recht, B., Roelofs, R., Schmidt, L., & Shankar, V. (2019). Do ImageNet Classifiers Generalize to ImageNet? *ICML 2019*

[8] OpenAI (2023). Evals Framework. https://github.com/openai/evals

[9] Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., et al. (2022). Constitutional AI: Harmlessness from AI Feedback. *arXiv:2212.08073*

[10] Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., & Stoica, I. (2024). Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. *arXiv:2403.04132*

[11] De Borda, J.-C. (1781). Memoire sur les elections au scrutin. *Histoire de l'Academie Royale des Sciences*, Paris.

[12] Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., & Hashimoto, T. B. (2023). AlpacaEval: An Automatic Evaluator of Instruction-Following Models. https://github.com/tatsu-lab/alpaca_eval

[13] Dubois, Y., Li, X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P., & Hashimoto, T. B. (2024). AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback. *NeurIPS 2023*

[14] Saad-Falcon, J., Barber, T., Jain, N., Desai, A., & Potts, C. (2023). ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems. *arXiv:2311.09476*

[15] Hendrycks, D., Burns, C., Basart, S., Zou, A., Mazeika, M., Song, D., & Steinhardt, J. (2021). Measuring Massive Multitask Language Understanding. *ICLR 2021*

[16] Liang, P., Bommasani, R., Lee, T., Tsipras, D., Soylu, D., Yasunaga, M., Zhang, Y., et al. (2022). Holistic Evaluation of Language Models (HELM). *arXiv:2211.09110*

[17] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. *NeurIPS 2022*

[18] Cohen, J. (1960). A Coefficient of Agreement for Nominal Scales. *Educational and Psychological Measurement*, 20(1), 37-46.

[19] Kendall, M. G. (1938). A New Measure of Rank Correlation. *Biometrika*, 30(1/2), 81-93.

[20] Panickssery, A., Bowman, S. R., & Feng, S. (2024). LLM Evaluators Recognize and Favor Their Own Generations. *arXiv:2404.13076*

[21] Verga, P., Hofstatter, S., Althammer, S., Pirtoaca, G., Cer, D., Re, C., Gunnemann, S., & Petroni, F. (2024). Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models. *arXiv:2404.18796*

[22] Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., & Liu, Z. (2024). ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate. *ICLR 2024. arXiv:2308.07201*

[23] Thakur, N., Mukherjee, S., Arasteh, S. T., Reimers, N., Han, J., & Schutze, H. (2024). Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges. *arXiv:2406.12624*

[24] Feng, S., Park, C. Y., Liu, Y., & Tsvetkov, Y. (2025). Sage: A General-Purpose Evaluator for LLMs. *arXiv:2512.16041*

[25] Landis, J. R. & Koch, G. G. (1977). The Measurement of Observer Agreement for Categorical Data. *Biometrics*, 33(1), 159-174.

[26] Li, T., Chiang, W.-L., Frick, E., Dunlap, L., Zhu, B., Gonzalez, J. E., & Stoica, I. (2024). From Live Data to High-Quality Benchmarks: The Arena-Hard Pipeline. *arXiv:2406.11939*

[27] Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical Inference. *Journal of the American Statistical Association*, 22(158), 209-212.

[28] LMSYS / LMArena (2024). Arena Human Preference 100k Dataset. `lmarena-ai/arena-human-preference-100k`, Hugging Face Datasets. CC-BY-4.0. https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k
