# Statistical Testing — Llama-3.2-3B Evaluation Results

**Date:** October 27, 2026 · **Branch:** `btt_hh`

**Runs analyzed** — both `meta-llama/Llama-3.2-3B-Instruct`, 4-bit QLoRA, Tesla T4, 500 samples, evaluated on the **same 4 test clusters × 5 prompts (n=20 predictions)**:

| | Run A | Run B |
|---|---|---|
| Run ID | `20260914_0241_Llama-3.2-3B-Instruct_ep3` | `20260929_2353_Llama-3.2-3B-Instruct_ep5` |
| Epochs | 3 | 5 |
| LoRA rank | 16 | 16 |
| Judge mode | reference | reference |

## Results summary

| Test | Result | p | Significant at α=0.05 |
|---|---|---:|---|
| Fine-tuned vs baseline latency, Run B | −0.129 s/label | 0.0023 | yes |
| Fine-tuned vs baseline latency, Run A | −0.146 s/label | 0.0192 | yes |
| Fine-tuned vs baseline latency, pooled (n=40) | −0.137 s/label | 0.0002 | yes |
| Fine-tuned vs teacher latency, Run B | −0.166 s/label | 0.0136 | yes |
| Fine-tuned vs baseline judge composite, Run B | +0.350 | 0.114 | no |
| Fine-tuned vs baseline judge composite, Run A | +0.150 | 0.342 | no |
| Run B vs Run A fine-tuned composite | +0.033 | 0.864 | no |
| Run B vs Run A baseline composite | −0.167 | 0.397 | no |

## 1. Latency

### Run B (5 epochs), test split, n=20 per model

| Model | Mean | SD | p95 | Throughput |
|---|---:|---:|---:|---:|
| Teacher (Claude Haiku) | 0.7705 s | 0.3611 | 1.1722 | — |
| Baseline | 0.7336 s | 0.1641 | 0.9720 | 81.8/min |
| Fine-tuned | 0.6046 s | 0.1002 | 0.7607 | 99.2/min |

Welch's ANOVA: F = 5.629, df2 = 32.8, **p = 0.0079**. Kruskal-Wallis: H = 10.871, p = 0.0044. Levene's test for equal variance: p = 0.527.

Paired comparisons (same 20 items):

| Comparison | Mean diff | Wilcoxon p | Holm-adj p | Win rate |
|---|---:|---:|---:|---:|
| Fine-tuned vs baseline | −0.1289 s | 0.0023 | 0.0157 | 16/20 (80%) |
| Fine-tuned vs teacher | −0.1658 s | 0.0136 | 0.121 | 15/20 (75%) |
| Baseline vs teacher | −0.0369 s | 0.674 | 0.681 | — |

Bootstrap 95% CI for the fine-tuned vs baseline speedup (10,000 resamples): **[−0.1974, −0.0611]**, excludes zero.

### Run A (3 epochs), test split, n=20 per model

| Model | Mean | SD | p95 |
|---|---:|---:|---:|
| Teacher | 0.7887 s | 0.2005 | 1.0213 |
| Baseline | 0.9127 s | 0.2277 | 1.3549 |
| Fine-tuned | 0.7669 s | 0.1725 | 0.9755 |

Welch's ANOVA: F = 2.726, df2 = 37.5, p = 0.0785. Kruskal-Wallis: H = 6.534, p = 0.0381.

Paired fine-tuned vs baseline: **−0.1459 s, Wilcoxon p = 0.0192**, faster on 15/20 (75%).

### Pooled across both runs

n = 40 paired observations: mean speedup **−0.1374 s**, **Wilcoxon p = 0.000206**, fine-tuned faster on **31/40 (78%)**.

### Measurement caveat

Teacher latency includes API and network overhead; student latency is local Colab GPU inference. Teacher timings also come from 105 calls across all splits (mean 0.7278 s, SD 0.2115, max 2.2331 s) while student timings cover only the 20 test items. The fine-tuned vs baseline comparison is the controlled one: identical hardware, identical inputs, identical call counts.

## 2. Quality — Run B (5 epochs)

Paired Wilcoxon signed-rank, Holm-corrected within each metric family.

### LLM judge

| Metric | Baseline | Fine-tuned | Gain | Win rate | p | Holm-adj p |
|---|---:|---:|---:|---:|---:|---:|
| equivalence | 3.300 | 3.850 | +0.550 | 69% | 0.102 | 0.408 |
| faithfulness | 4.150 | 4.500 | +0.350 | 80% | 0.131 | 0.408 |
| specificity | 3.450 | 3.600 | +0.150 | 58% | 0.465 | 0.465 |
| composite | 3.633 | 3.983 | +0.350 | 63% | 0.114 | 0.408 |

### Similarity metrics

| Metric | Baseline | Fine-tuned | Gain | Win rate | p | Holm-adj p |
|---|---:|---:|---:|---:|---:|---:|
| cosine (same) | 0.8061 | 0.8422 | +0.0361 | 63% | 0.243 | 0.730 |
| cosine (multi) | 0.8483 | 0.8991 | +0.0508 | 74% | 0.027 | 0.134 |
| ROUGE-L (same) | 0.5017 | 0.5593 | +0.0576 | 44% | 0.679 | 1.000 |
| ROUGE-L (multi) | 0.5750 | 0.7307 | +0.1557 | 82% | 0.015 | 0.089 |
| BERTScore (same) | 0.9134 | 0.9221 | +0.0087 | 50% | 0.674 | 1.000 |
| BERTScore (multi) | 0.9266 | 0.9425 | +0.0159 | 60% | 0.114 | 0.456 |

All 10 metrics moved in the positive direction. None reaches significance after correction. Bootstrap 95% CI for the composite gain: **[−0.167, +0.833]**, includes zero.

## 3. Quality — Run A (3 epochs)

| Metric | Baseline | Fine-tuned | Gain | p | Holm-adj p |
|---|---:|---:|---:|---:|---:|
| equivalence | 3.450 | 3.700 | +0.250 | 0.272 | 1.000 |
| faithfulness | 4.350 | 4.450 | +0.100 | 0.414 | 1.000 |
| specificity | 3.600 | 3.700 | +0.100 | 0.527 | 1.000 |
| composite | 3.800 | 3.950 | +0.150 | 0.342 | 1.000 |
| cosine (same) | 0.8211 | 0.8199 | −0.0012 | 0.748 | 1.000 |
| cosine (multi) | 0.8795 | 0.8760 | −0.0035 | 0.872 | 1.000 |
| ROUGE-L (same) | 0.5029 | 0.5003 | −0.0025 | 0.936 | 1.000 |
| ROUGE-L (multi) | 0.6083 | 0.6318 | +0.0235 | 0.334 | 1.000 |
| BERTScore (same) | 0.9145 | 0.9073 | −0.0071 | 0.701 | 1.000 |
| BERTScore (multi) | 0.9329 | 0.9293 | −0.0036 | 0.701 | 1.000 |

Judge dimensions improved; four of six similarity metrics decreased. No result is significant.

## 4. Direct comparison of the two runs

Both runs scored the same 20 items, so these are paired comparisons.

| Measure | Run A (ep3) | Run B (ep5) | Diff | Wilcoxon p |
|---|---:|---:|---:|---:|
| Baseline composite | 3.800 | 3.633 | −0.167 | 0.397 |
| Fine-tuned composite | 3.950 | 3.983 | +0.033 | 0.864 |
| Gain from fine-tuning | +0.150 | +0.350 | +0.200 | 0.569 |
| Latency speedup | −0.1459 s | −0.1289 s | — | — |

The baseline models in both runs are the same un-fine-tuned checkpoint evaluated on the same 20 items. Their composite scores differ by **0.167 points**.

## 5. Statistical power (Run B)

Measured SD of the per-item composite gain: **1.177** on a 1–5 scale.

- Achieved power at n=20: **0.244**
- Sample size required to detect the observed +0.350 gain at 80% power: **n ≈ 91**

Minimum detectable gain at 80% power, using the measured SD:

| n | 20 | 35 | 50 | 75 | 100 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| Detectable gain | +0.78 | +0.57 | +0.48 | +0.39 | +0.33 | +0.23 |

Per judge dimension:

| Dimension | Gain | SD | dz | n required |
|---|---:|---:|---:|---:|
| equivalence | +0.550 | 1.538 | 0.358 | 64 |
| faithfulness | +0.350 | 0.988 | 0.354 | 65 |
| composite | +0.350 | 1.177 | 0.297 | 91 |
| specificity | +0.150 | 1.226 | 0.122 | 527 |

## 6. Per-prompt breakdown (Run B, n=4 each)

| Prompt | Baseline | Fine-tuned | Gain |
|---|---:|---:|---:|
| P1 | 3.42 | 4.00 | +0.58 |
| P2 | 3.33 | 4.08 | +0.75 |
| P3 | 3.92 | 4.00 | +0.08 |
| P4 | 3.42 | 4.17 | +0.75 |
| P5 | 4.08 | 3.67 | −0.42 |

P5 is the only prompt that regressed and had the highest baseline score. n=4 per prompt; no per-prompt test was run.

## 7. Threshold analysis (Run B)

| Threshold | Baseline pass | Fine-tuned pass | McNemar p |
|---|---:|---:|---:|
| ≥ 3.5 | 11/20 (55%) | 12/20 (60%) | 1.000 |
| ≥ 4.0 | 7/20 (35%) | 11/20 (55%) | 0.289 |
| ≥ 4.5 | 3/20 (15%) | 6/20 (30%) | 0.453 |

Precision/recall/F1 against the teacher could not be computed: in reference mode the judge assigns the teacher a fixed 5.0, leaving no variation.

## 8. Unit of analysis (Run B)

| Unit | n | Mean gain | SD | Wilcoxon p |
|---|---:|---:|---:|---:|
| Per prediction | 20 | +0.350 | 1.177 | 0.114 |
| Per cluster (prompts averaged) | 4 | +0.350 | 0.476 | 0.375 |

Cluster-level results: fine-tuned higher on 3 of 4 clusters.

## 9. Cost (Run B, from `business_eval.csv`)

| | Teacher | Fine-tuned SLM |
|---|---:|---:|
| Cost per label | $0.000130 | $0 (Colab) |
| 105 labels | $0.01365 | $0 |
| Throughput | — | 99.2/min |
| Fine-tuning wall time | — | 3.34 min |

The $0 SLM figure reflects Colab compute only and excludes production GPU hosting.

## Methods

| Comparison | Test | Rationale |
|---|---|---|
| Baseline vs fine-tuned | Wilcoxon signed-rank | Paired data; n=20; bounded, discrete, skewed scores |
| Three-model latency | Welch's ANOVA + Kruskal-Wallis | Does not assume equal variance |
| Pairwise latency | Welch's t + Holm | Unequal variance; corrects 3 comparisons |
| Threshold pass/fail | McNemar exact binomial | Paired binary outcomes, small counts |
| Key effects | Bootstrap 95% CI, 10,000 resamples | Distribution-free confirmation |

Holm-Bonferroni applied within each metric family. scipy 1.13.1.

---

Produced with `analysis/stats_tests.py`. Reproduce: `python analysis/stats_tests.py --run_dir <run_dir> --split test`
