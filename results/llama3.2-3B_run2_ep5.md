# Experiment Results Summary — Llama-3.2-3B-Instruct (Run 2, 5 epochs)

## Model / Setup

- **Student model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)
- **Task:** SLM distillation — training the student to match teacher outputs
- **Run ID:** `20260929_2353_Llama-3.2-3B-Instruct_ep5`

### Config changes vs. Run 1

Run 2 overrode these values in Colab (Run 1 assumed to use `phase1_config.yaml` defaults apart from the model):

| Setting | Run 1 | Run 2 |
|---|---|---|
| `num_train_epochs` | 3 | **5** |
| `learning_rate` | 2e-4 | **1e-4** |
| `lora.r` / `lora_alpha` | 16 / 16 | **32 / 32** |
| `evaluation.run_bertscore` | false | **true** (eval only) |

## Dataset

- Evaluation performed on **test split** only (train/val metrics not recorded for this run).
- Teacher model serves as the gold-standard reference for all similarity and judge scores.

## Training Results

> Training details (final loss, runtime) were not captured for this run. Add them here from the run log / Colab config for future reference.

## Baseline vs. Fine-Tuned Evaluation

Cosine similarity measures how closely each model's output embeddings match the teacher's. BERTScore F1 is a token-level semantic similarity metric (new in this run).

| Model | Cosine (same) | Cosine (multi) | ROUGE-L (same) | ROUGE-L (multi) | BERTScore F1 (same) | BERTScore F1 (multi) |
|---|---|---|---|---|---|---|
| Teacher | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| Baseline | 0.806 | 0.848 | 0.502 | 0.575 | 0.913 | 0.927 |
| Fine-tuned | 0.842 | 0.899 | 0.559 | 0.731 | 0.922 | 0.943 |
| **Δ (FT − Base)** | **+0.036** | **+0.051** | **+0.058** | **+0.156** | **+0.009** | **+0.016** |

Fine-tuning improved every metric. The largest gain is ROUGE-L in the multi-reference setting (+0.156), and cosine similarity now shows a clear improvement (+0.036 / +0.051), unlike Run 1 where it was essentially flat.

## LLM-as-a-Judge Results

Scores are on a 1–5 scale across three dimensions, with composite being the mean.

| Model | Equivalence | Faithfulness | Specificity | Composite |
|---|---|---|---|---|
| Teacher | 5.00 | 5.00 | 5.00 | 5.00 |
| Baseline | 3.30 | 4.15 | 3.45 | 3.63 |
| Fine-tuned | 3.85 | 4.50 | 3.60 | 3.98 |
| **Δ (FT − Base)** | **+0.55** | **+0.35** | **+0.15** | **+0.35** |

Fine-tuning improved all three dimensions, led by equivalence (+0.55), for a composite gain of **+0.35** (vs. +0.27 in Run 1).

## Comparison with Run 1

| Metric (fine-tuned, test) | Run 1 (ep3) | Run 2 (ep5) | Change |
|---|---|---|---|
| Cosine (same) | 0.819 | 0.842 | +0.023 |
| Cosine (multi) | 0.877 | 0.899 | +0.022 |
| ROUGE-L (same) | 0.473 | 0.559 | +0.086 |
| ROUGE-L (multi) | 0.610 | 0.731 | +0.121 |
| Judge equivalence | 3.60 | 3.85 | +0.25 |
| Judge faithfulness | 4.55 | 4.50 | −0.05 |
| Judge specificity | 3.60 | 3.60 | 0.00 |
| Judge composite | 3.92 | 3.98 | +0.06 |

Note: baseline scores also differ between runs (e.g. cosine same 0.820 → 0.806, ROUGE-L same 0.466 → 0.502), even though the baseline model is unchanged. This suggests generation is non-deterministic (sampling) or the test sample differs between runs, so part of the run-to-run difference may be noise. The **within-run** Δ (fine-tuned vs. baseline) is the more reliable comparison: Run 1 Δ cosine same was −0.001, Run 2 is +0.036.

## Conclusion

The combined changes (5 epochs, lower LR of 1e-4, LoRA rank 32) turned LoRA fine-tuning from a marginal effect into a **clear, consistent improvement** across all similarity metrics. Where Run 1 showed gains mainly in the LLM-judge scores with flat embedding similarity, Run 2 shows the student moving measurably closer to the teacher in cosine similarity (+0.036 / +0.051), lexical overlap (ROUGE-L +0.058 / +0.156), and BERTScore (+0.009 / +0.016). The LLM judge agrees: composite rose from 3.63 to 3.98 (+0.35), driven mostly by equivalence (+0.55).

The large multi-reference ROUGE-L gain suggests the fine-tuned model is adopting the teacher's phrasing and label vocabulary. Compared with Run 1, the fine-tuned judge scores are only slightly higher (composite 3.92 → 3.98), and specificity did not improve at all — the extra epochs mainly helped the model match the teacher's wording and meaning rather than making labels more specific. Because three training settings changed at once, these gains can't be attributed to any single one; an ablation (e.g. 5 epochs with the original LR and rank) would isolate the effect of each. Next steps: capture training/validation loss to check for overfitting at 5 epochs, fix generation seeds (or use greedy decoding) so baselines are comparable across runs, and try additional epochs or a higher LoRA rank to see where gains plateau.
