# Phi-3.5 Mini Smoke Test Results

## Experiment

- Author: Harinandan Nisheed
- Date: September 13, 2026
- Model: microsoft/Phi-3.5-mini-instruct
- Environment: Google Colab, NVIDIA T4
- Dataset: 500 Bitext support tickets configured
- Training: 3 epochs, 4-bit QLoRA
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Pipeline exit code: 0

## Completed Work

Completed the Phase 1 pipeline, including clustering, preprocessing,
teacher labeling, fine-tuning, baseline evaluation, fine-tuned evaluation,
LLM judging, and business evaluation.

Saved the adapter, training checkpoints, configuration, and evaluation
files to Google Drive. Pushed the notebook, configuration, and result
summaries to the btt_setup_hn branch.

## Test Results

Higher similarity and judge scores are better. Lower latency is better.
Similarity scores are not accuracy percentages.

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Cosine similarity, same prompt | 0.793341 | 0.809278 |
| Cosine similarity, multi-reference | 0.872364 | 0.891755 |
| ROUGE-L, same prompt | 0.541147 | 0.565275 |
| ROUGE-L, multi-reference | 0.713991 | 0.699896 |
| BERTScore F1, same prompt | 0.924928 | 0.929018 |
| BERTScore F1, multi-reference | 0.948468 | 0.951208 |
| Judge equivalence / 5 | 3.50 | 3.40 |
| Judge faithfulness / 5 | 4.40 | 4.30 |
| Judge specificity / 5 | 3.65 | 3.60 |
| Judge composite / 5 | 3.8500 | 3.7667 |
| Mean inference latency, seconds | 0.685575 | 0.660370 |
| Throughput, labels per minute | 87.517779 | 90.858089 |

## Training and Cost

- Fine-tuning time: 153.47 seconds, approximately 2.56 minutes.
- Reported teacher labeling calls: 105.
- Estimated teacher labeling cost: $0.01365.
- Reported student inference cost: $0 under the free Colab assumption.
- These figures do not establish total experiment or deployment cost.
  The teacher labeling estimate does not necessarily include judge calls.

## Findings

Fine-tuning modestly improved cosine similarity, BERTScore, and
same-prompt ROUGE-L. Multi-reference ROUGE-L and all three judge
dimensions decreased slightly.

Mean student inference latency decreased by approximately 3.7% in this
run. These mixed results verify pipeline execution but do not establish
a consistent quality or speed improvement across repeated experiments.

## Limitations

- Evaluation was restricted to the test split. Train and validation
  metric rows contain NaN because those splits were not evaluated.
- Teacher similarity scores compare the teacher against its own labels.
- Reference-mode teacher judge scores are hardcoded at 5/5.
- The saved adapter's base_model_name_or_path is null.
  The run configuration and training report identify Phi.
- Reloading the saved adapter in a fresh session remains untested.

## Files

- [Metric summary](metrics_summary.csv)
- [Judge summary](judge_summary.csv)
- [Business evaluation](business_eval.csv)
- [Experiment configuration](phase1_phi_colab.yaml)
- [Notebook](../../notebooks/Hari_Phi_Smoke_Test.ipynb)
