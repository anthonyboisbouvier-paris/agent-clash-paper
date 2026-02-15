# Agent Clash — Validation Data & Article

This repository contains the validation data, raw results, and article for the Agent Clash multi-judge evaluation framework.

**Paper:** *Multi-Agent Judging for LLM Evaluation: Concordance with Human Preferences Across Capability Gaps*

**Authors:** Anthony Boisbouvier, Claude Opus 4.5 (Anthropic)

**Production platform:** [agent-clash.ai](https://www.agent-clash.ai/)
**Production codebase:** [github.com/anthonyboisbouvier-paris/llm-arena](https://github.com/anthonyboisbouvier-paris/llm-arena)

---

## Repository Structure

```
agent-clash-paper/
  article/
    en/article_v4.md          # English article
    fr/article_v4_fr.md       # French translation
  validation/
    workflows/
      validation_pipeline_workflow.json   # n8n validation workflow (exported)
    data/
      mt_bench_100_evaluations.json       # MT-Bench sample (N=100)
      battles_100.json                    # Arena sample (N=100)
    results/
      validation_results.json             # Exp 1: MT-Bench results
      arena_validation_results.json       # Exp 2: Arena Run 1
      arena_validation_results_run2.json  # Exp 2: Arena Run 2
      arena_validation_results_run3.json  # Exp 2: Arena Run 3 (targeted)
      e2e_mini_results.json              # End-to-end stability test
      judge_matrix_data.json             # Per-judge voting matrices
```

## Key Results

| Experiment | Dataset | Concordance | Cohen's kappa | Test-retest |
|---|---|---|---|---|
| Exp 1 | MT-Bench (6 models, large gaps) | **88.0%** | 0.760 | — |
| Exp 2 | Arena (25 frontier models) | **76.0%** | 0.520 | **91.0%** |

## Datasets

Both datasets are used under the **CC-BY-4.0** license:

- **MT-Bench Human Judgments** — [lmsys/mt_bench_human_judgments](https://huggingface.co/datasets/lmsys/mt_bench_human_judgments)
- **Arena Human Preference 100k** — [lmarena-ai/arena-human-preference-100k](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-100k)

## License

This work is provided for academic and research purposes. The validation datasets retain their original CC-BY-4.0 license.
