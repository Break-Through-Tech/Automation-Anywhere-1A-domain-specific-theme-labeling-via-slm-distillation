# Experiment Results Summary — Llama-3.2-3B-Instruct (Run 1)

## Model / Setup

- **Student model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-tuning method:** LoRA (Low-Rank Adaptation)
- **Task:** SLM distillation — training the student to match teacher outputs
- **Run ID:** `20260915_0733_Llama-3.2-3B-Instruct_ep3`
- **Estimated runtime:** ~13 minutes

## Dataset

- Evaluation performed on **test split** only (train/val metrics not recorded for this run).
- Teacher model serves as the gold-standard reference for all similarity and judge scores.

## Training Results

> Training details (final loss, epoch count, dataset split sizes) were not captured for this run. Add them here from the run log for future reference.

## Baseline vs. Fine-Tuned Evaluation (Cosine Similarity)

Cosine similarity measures how closely each model's output embeddings match the teacher's.

| Model | Same-language sim | Multi-language sim | ROUGE-L (same) | ROUGE-L (multi) |
|---|---|---|---|---|
| Teacher | 1.000 | 1.000 | 1.000 | 1.000 |
| Baseline | 0.820 | 0.872 | 0.466 | 0.584 |
| Fine-tuned | 0.819 | 0.877 | 0.473 | 0.610 |

Fine-tuning produced a marginal **drop** in same-language cosine similarity (−0.001) but a small **gain** in multi-language similarity (+0.005). ROUGE-L improved slightly in both conditions, more noticeably in the multi-language setting (+0.026).

## LLM-as-a-Judge Results

Scores are on a 1–5 scale across three dimensions, with composite being the mean.

| Model | Equivalence | Faithfulness | Specificity | Composite |
|---|---|---|---|---|
| Teacher | 5.00 | 5.00 | 5.00 | 5.00 |
| Baseline | 3.30 | 4.30 | 3.35 | 3.65 |
| Fine-tuned | 3.60 | 4.55 | 3.60 | 3.92 |

Fine-tuning improved all three dimensions: equivalence (+0.30), faithfulness (+0.25), and specificity (+0.25), for a composite gain of **+0.27**.

## Conclusion

Fine-tuning with LoRA produced **modest but consistent improvements** over the baseline, most clearly reflected in the LLM-as-a-Judge scores. The composite score rose from 3.65 to 3.92, with gains across all evaluated dimensions. Embedding-based cosine similarity was nearly unchanged, suggesting the model's semantic direction stayed similar while the quality and specificity of outputs improved in ways that surface-level similarity metrics don't fully capture.

The improvements are real but limited — a 3-point model fine-tuned for ~13 minutes on a distillation task is working against a capacity ceiling. The multi-language gains in both cosine sim and ROUGE-L hint that the fine-tuning helped generalization beyond the same-language distribution. Next steps would be to record training loss curves and increase epochs or dataset size to see if further fine-tuning narrows the gap with the teacher.
