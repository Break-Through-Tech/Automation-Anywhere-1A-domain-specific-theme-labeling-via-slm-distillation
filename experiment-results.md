# Gemma 3 4B Smoke Test Results

## Model

- Student model: `google/gemma-3-4b-it`
- Fine-tuning method: QLoRA / LoRA
- GPU: Tesla T4
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05

## Dataset

- Total tickets: 497
- Total clusters: 21
- Cluster split — train: 14, val: 3, test: 4
- Training epochs: 3

## Training Result

- Training completed successfully
- Fine-tuning time: 3.7 min
- Model loading time: 42.1s (fine-tuned), 45.8s (baseline)
- Fine-tuned adapter was saved to Google Drive
- Run ID: `20260915_0252_gemma-3-4b-it_ep3`

## Baseline vs Fine-Tuned Evaluation

### Cosine Similarity (test split)

| Model | Same-cluster Similarity | Multi-cluster Similarity |
|---|---:|---:|
| Teacher (Claude Haiku) | 1.0000 | 1.0000 |
| Baseline SLM | 0.8041 | 0.8679 |
| Fine-tuned SLM | 0.8386 | 0.8874 |

Average improvement (same-cluster): **+0.0345**

### ROUGE-L (test split)

| Model | Same-cluster | Multi-cluster |
|---|---:|---:|
| Teacher | 1.0000 | 1.0000 |
| Baseline SLM | 0.5046 | 0.6048 |
| Fine-tuned SLM | 0.5354 | 0.6393 |

### LLM-as-a-Judge (test split)

| Metric | Teacher | Baseline SLM | Fine-tuned SLM |
|---|---:|---:|---:|
| Faithfulness | 5.00 | 4.20 | 4.45 |
| Specificity | 5.00 | 3.55 | 3.65 |
| Equivalence | 5.00 | 3.30 | 3.55 |
| Composite | 5.00 | 3.68 | 3.89 |

*(Win/tie/loss counts per example were not part of this pipeline's output — only aggregate scores.)*

## Business Evaluation

| Metric | Teacher (Claude Haiku) | Student SLM (Gemma 3 4B) |
|---|---:|---:|
| Mean latency | 0.80s / call | 1.781s / label (fine-tuned) |
| Total calls | 105 | — |
| Estimated cost | $0.0137 USD | ~$0.00 (local/Colab) |
| Speed vs. teacher | — | 0.5x faster |

## Conclusion

The fine-tuned Gemma 3 4B model improved on every quality metric compared with its untrained baseline — cosine similarity, ROUGE-L, and all three LLM-judge dimensions (faithfulness, specificity, equivalence). It remains well below teacher-level quality, which is expected for a short, 3-epoch smoke test on a small dataset. The main practical advantage is cost: once fine-tuned, inference is effectively free to run locally, compared to per-call API costs for the teacher LLM.

## Notes / Issues Hit

Hit a bug during dataset construction: `AttributeError: 'Gemma3Processor' object has no attribute 'encode'` in `dataset.py` (line ~194). Gemma 3 loads with a Processor object rather than a plain tokenizer, which the pipeline didn't originally account for. Patched locally by calling `.tokenizer.encode()` when a `.tokenizer` attribute is present, and pushed the fix to branch `btt_setup_mt`. Other teammates fine-tuning Gemma-family models will likely hit the same error.

# Qwen3 8B Smoke Test Results

## Status: BLOCKED — Out of Memory on T4

## Model
- Student model: `Qwen/Qwen3-8B`
- Fine-tuning method: QLoRA / LoRA (4-bit)
- GPU: Tesla T4 (14.56 GB VRAM)

## What happened
Pipeline successfully loaded the model and began fine-tuning setup, but failed with `torch.OutOfMemoryError` during LoRA adapter injection. The pipeline appears to load the full model into GPU memory twice (once during dataset/tokenizer setup, again during actual fine-tuning) without freeing the first copy. This fits for smaller models (e.g. Gemma 3 4B) but exceeds the T4's 14.5GB budget for an 8B model.

## Fixes attempted
- Forced `device_map={"": 0}` to prevent automatic CPU/disk offload (fixed the first error)
- Reduced `per_device_train_batch_size` from 4 → 1, increased `gradient_accumulation_steps` to compensate
- Reduced `max_seq_length` from 2048 → 1024
- Restarted Colab runtime to clear stale GPU memory

Still hit `CUDA out of memory` — tried to allocate a final 2 MiB with 14.39/14.56 GB already in use.

## Conclusion
Qwen3 8B, as currently implemented in this pipeline, does not fit on a free-tier Colab T4 GPU.

# Qwen3 8B Smoke Test Results

## Status: RESOLVED — Fixed via code change after optimization

## Model

- Student model: `Qwen/Qwen3-8B`
- Fine-tuning method: QLoRA / LoRA (4-bit)
- GPU: Tesla T4 (14.56 GB VRAM)
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05

## Dataset

- Total tickets: 497
- Total clusters: 21
- Cluster split — train: 14, val: 3, test: 4
- Training epochs: 3

## Initial Blocker

Pipeline successfully loaded the model and began fine-tuning setup, but failed with `torch.OutOfMemoryError` during LoRA adapter injection. The pipeline was loading the full model into GPU memory twice (once during dataset/tokenizer setup, again during actual fine-tuning) without freeing the first copy. This fit for smaller models (e.g. Gemma 3 4B) but exceeded the T4's 14.5GB budget for an 8B model.

## Fixes Attempted Before Finding the Root Cause

- Forced `device_map={"": 0}` to prevent automatic CPU/disk offload
- Reduced `per_device_train_batch_size` from 4 → 1, increased `gradient_accumulation_steps` to compensate
- Reduced `max_seq_length` from 2048 → 1024
- Restarted Colab runtime to clear stale GPU memory

Still hit `CUDA out of memory` — tried to allocate a final 2 MiB with 14.39/14.56 GB already in use.

## Root Cause and Code Fix

Traced the issue to `pipeline.py`: a temporary model was loaded just to build the tokenizer for dataset construction, then discarded without freeing GPU memory. When the real training model loaded afterward, both copies competed for VRAM. Added an explicit delete + GPU cache clear for the temporary model before the real load begins, ensuring only one copy of the model is ever in memory at a time. Batch size and sequence length were restored to original values once the root cause was fixed.

## Training Result

- Training completed successfully after the fix — no memory errors
- Fine-tuning time: 4.3 min
- Model loading time: 63.1s (fine-tuned), 60.4s (baseline)
- Fine-tuned adapter was saved to Google Drive
- Run ID: `20260929_1234_Qwen3-8B_ep3`

## Baseline vs Fine-Tuned Evaluation

### Cosine Similarity (test split)

| Model | Same-cluster Similarity | Multi-cluster Similarity |
|---|---:|---:|
| Teacher (Claude Haiku) | 1.0000 | 1.0000 |
| Baseline SLM | 0.4333 | 0.4636 |
| Fine-tuned SLM | 0.3406 | 0.3676 |

Change (same-cluster): **-0.0927**

### ROUGE-L (test split)

| Model | Same-cluster | Multi-cluster |
|---|---:|---:|
| Teacher | 1.0000 | 1.0000 |
| Baseline SLM | 0.1136 | 0.1323 |
| Fine-tuned SLM | 0.1073 | 0.1270 |

### LLM-as-a-Judge (test split)

| Metric | Teacher | Baseline SLM | Fine-tuned SLM |
|---|---:|---:|---:|
| Equivalence | 5.00 | 2.90 | 3.95 |
| Faithfulness | 5.00 | 3.15 | 4.30 |
| Specificity | 5.00 | 2.70 | 3.60 |
| Composite | 5.00 | 2.92 | 3.95 |

## Business Evaluation

| Metric | Teacher (Claude Haiku) | Student SLM (Qwen3 8B) |
|---|---:|---:|
| Mean latency | 0.80s / call | 5.434s / label (fine-tuned) |
| Total calls | 105 | — |
| Estimated cost | $0.0137 USD | ~$0.00 (local/Colab) |
| Speed vs. teacher | — | 0.1x faster |

## Conclusion

After fixing the underlying memory bug, Qwen3 8B trained successfully on a free-tier Colab T4 GPU. Results were mixed: the LLM-judge composite score improved substantially after fine-tuning (2.92 → 3.95), suggesting the model learned to produce labels judged as more faithful and specific. However, cosine similarity and ROUGE-L against the reference labels slightly decreased, suggesting fine-tuning shifted label phrasing away from the teacher's exact wording even as an LLM judge rated the results as higher quality. Inference latency was notably higher than Gemma 3 4B, consistent with Qwen3 8B being a larger model.

## Notes / Issues Hit

Hit a `CUDA out of memory` error during the first fine-tuning attempt, caused by the pipeline loading the model into GPU memory twice without releasing the first copy. Traced and fixed the root cause in `pipeline.py` rather than only working around it with reduced batch size — this fix should also help other 7B–9B class models (Mistral-7B, Gemma 2 9B) avoid the same memory ceiling. Pushed the fix to branch `btt_setup_mt`.
