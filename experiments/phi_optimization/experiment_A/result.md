# Phi-3.5 Mini Optimization: Experiment A

## Outcome

The fine-tuned model achieved a validation LLM judge composite of
**4.0444/5**, exceeding the target of 4/5.

The baseline scored **4.0000/5** on the same validation split.
The improvement was **0.0444 points**, so this is a small observed
gain, not evidence of a large or statistically reliable improvement.

Training and evaluation completed with pipeline exit code 0.

## Experiment setup

- Date: September 29, 2026
- Author: Harinandan Nisheed
- Model: microsoft/Phi-3.5-mini-instruct
- GPU: NVIDIA Tesla T4
- Method: 4-bit QLoRA through standard Hugging Face / PEFT
- Dataset: 497 saved Bitext tickets across 21 clusters
- Existing teacher labels reused without regeneration
- Split strategy: cluster-level
- Validation evaluation: 3 clusters, 5 prompts each, 15 predictions
- Epochs: 3
- Learning rate: 0.0002
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Training batch size: 1
- Gradient accumulation: 16
- Training precision: FP16
- Maximum sequence length: 2048
- Run ID: 20260929_2327_Phi-3.5-mini-instruct_ep3
- Source commit before publishing these changes: f79e4f5132723f0206b385f22c2cb9d479da9f3c

## Validation results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Judge composite / 5 | 4.0000 | 4.0444 |
| Equivalence / 5 | 3.6667 | 3.8000 |
| Faithfulness / 5 | 4.5333 | 4.5333 |
| Specificity / 5 | 3.8000 | 3.8000 |
| Cosine similarity, same prompt | 0.7727 | 0.7527 |
| Cosine similarity, multi-reference | 0.8354 | 0.8714 |
| ROUGE-L, same prompt | 0.5029 | 0.4995 |
| ROUGE-L, multi-reference | 0.6599 | 0.6620 |
| BERTScore F1, same prompt | 0.9163 | 0.9158 |
| BERTScore F1, multi-reference | 0.9370 | 0.9408 |
| Mean inference latency, seconds | 1.616 | 1.982 |

Values above are rounded from the log. The accompanying CSV files
contain the saved metric values.

Fine-tuning took approximately **4.7 minutes**. The complete pipeline
took approximately **25 minutes**.

## Code changes

### Pipeline memory management

- Replaced full model loading during dataset construction with
  tokenizer-only loading.
- Released baseline model and tokenizer references after inference.
- Released fine-tuned model, base-model, and tokenizer references
  before subsequent evaluation stages.

These changes address unnecessary model retention in GPU memory.
Peak-memory savings were not measured.

### Model loading and training

- Replaced the automatic Unsloth loading path in the Colab loader
  with standard Hugging Face loading.
- Loaded a clean base model before baseline inference.
- Applied LoRA during training and loaded the saved adapter separately
  for fine-tuned inference.
- Selected FP16 quantization computation on the T4 instead of
  hardcoding BF16.
- Used SDPA attention and explicit single-GPU placement.
- Prepared the quantized model with prepare_model_for_kbit_training
  before attaching LoRA.
- Used SFTConfig to pass max_length and dataset settings through the
  installed TRL API rather than dropping unsupported trainer arguments.
- Explicitly retained full-text training loss and disabled packing.
- Added base-model identification to adapter metadata when saving.

### Evaluation and reproducibility

- Added evaluation.inference_splits to select validation clusters.
- Enabled BERTScore alongside cosine similarity, ROUGE-L, LLM judging,
  and business evaluation.
- Set judge sampling high enough to include all validation predictions.
- Preserved a separate data snapshot, experiment configuration,
  code snapshot, and execution log in Drive.

## Interpretation

The above-4 validation target was met. The judge improvement came
from equivalence; faithfulness and specificity were unchanged.
Multi-reference similarity metrics improved, while same-prompt
metrics declined slightly. Fine-tuned inference was slower.

Several implementation changes were made together. This experiment
does not isolate which change caused the observed score difference.

The earlier smoke test scored approximately 3.77 on the TEST split
using a different implementation. That score cannot be directly
compared with this validation score as a measured improvement.

## Limitations

- This is a validation result, not a final test-set result.
- The 15 predictions come from only three clusters and are not
  15 independent ticket examples.
- Only one run is reported; judge variability was not quantified.
- The judge log reported retries with the temperature parameter
  removed, so deterministic judge settings were not confirmed.
- Teacher judge scores are hardcoded reference values.
- Teacher labeling was skipped. Zero teacher latency/cost fields
  represent unavailable or unmeasured values for this run.
- Judge API costs are not established by the teacher-labeling cost.
- A reported student inference cost of zero reflects the free Colab
  assumption, not zero deployment cost.
- All metric types ran on validation; train/test NaNs are intentional.

## Files

- [Quality metrics](metrics_summary.csv)
- [Judge scores](judge_summary.csv)
- [Business metrics](business_eval.csv)
- [Experiment configuration](phi_optimization_A.yaml)
- [Patched pipeline](../../../code/phase1/pipeline.py)
- [Patched trainer](../../../code/phase1/finetuning/trainer.py)

Model weights, checkpoints, and the execution log remain in Drive.
Experiment B is not included in this report.
