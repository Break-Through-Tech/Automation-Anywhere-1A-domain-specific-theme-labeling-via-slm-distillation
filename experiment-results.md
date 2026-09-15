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
