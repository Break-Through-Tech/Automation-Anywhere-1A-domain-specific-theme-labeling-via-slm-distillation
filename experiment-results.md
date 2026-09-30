# Gemma 3 4B Smoke Test Results
**Date: Sept 15, 2026**

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

## Business Evaluation
| Metric | Teacher (Claude Haiku) | Student SLM (Gemma 3 4B) |
|---|---:|---:|
| Mean latency | 0.80s / call | 1.781s / label (fine-tuned) |
| Total calls | 105 | — |
| Estimated cost | $0.0137 USD | ~$0.00 (local/Colab) |
| Speed vs. teacher | — | 0.5x faster |

## Conclusion
The fine-tuned Gemma 3 4B model improved on every quality metric compared with its untrained baseline. It remains well below teacher-level quality, expected for a short, 3-epoch smoke test on a small dataset.

## Notes / Issues Hit
Hit a bug during dataset construction: `AttributeError: 'Gemma3Processor' object has no attribute 'encode'` in `dataset.py` (line ~194). Gemma 3 loads with a Processor object rather than a plain tokenizer, which the pipeline didn't originally account for. Patched locally by calling `.tokenizer.encode()` when a `.tokenizer` attribute is present, and pushed the fix to branch `btt_setup_mt`.

---

# Qwen3 8B Smoke Test Results
**Date: Sept 15, 2026**

## Status: BLOCKED — Out of Memory on T4

## Model
- Student model: `Qwen/Qwen3-8B`
- Fine-tuning method: QLoRA / LoRA (4-bit)
- GPU: Tesla T4 (14.56 GB VRAM)

## What happened
Pipeline successfully loaded the model and began fine-tuning setup, but failed with `torch.OutOfMemoryError` during LoRA adapter injection. The pipeline appears to load the full model into GPU memory twice (once during dataset/tokenizer setup, again during actual fine-tuning) without freeing the first copy.

## Fixes attempted
- Forced `device_map={"": 0}` to prevent automatic CPU/disk offload (fixed the first error)
- Reduced `per_device_train_batch_size` from 4 → 1, increased `gradient_accumulation_steps` to compensate
- Reduced `max_seq_length` from 2048 → 1024
- Restarted Colab runtime to clear stale GPU memory

Still hit `CUDA out of memory` — tried to allocate a final 2 MiB with 14.39/14.56 GB already in use.

## Conclusion
Qwen3 8B, as implemented in this pipeline, does not fit on a free-tier Colab T4 GPU without further changes.

---

# Qwen3 8B Smoke Test Results — Unblocked
**Date: Sept 29, 2026**

## Status: RESOLVED — Fixed via code change

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
After fixing the underlying memory bug, Qwen3 8B trained successfully on a free-tier Colab T4 GPU. LLM-judge composite score improved substantially (2.92 → 3.95), but cosine similarity and ROUGE-L against the reference labels slightly decreased — suggesting fine-tuning shifted label phrasing away from the teacher's exact wording even as an LLM judge rated the results as higher quality.

## Notes / Issues Hit
Fixed root cause in `pipeline.py` rather than only working around it with reduced batch size — this fix should also help other 7B–9B class models (Mistral-7B, Gemma 2 9B) avoid the same memory ceiling. Pushed the fix to branch `btt_setup_mt`.

---

# Gemma 3 4B — Hyperparameter Experiment
**Date: Sept 29, 2026**

## Status: EXPERIMENT — Did not improve over baseline

## Model
- Student model: `google/gemma-3-4b-it`
- Fine-tuning method: QLoRA / LoRA
- GPU: Tesla T4
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05

## Change Tested
Attempted to push the composite LLM-judge score above 4.0 by increasing training epochs (3 → 6) and learning rate (2e-4 → 3e-4) simultaneously.

## Training Result
- Training completed successfully
- Fine-tuning time: 6.6 min
- Run ID: `20260929_2341_gemma-3-4b-it_ep6`

## Results vs. Original 3-Epoch Run

| Metric | Original (3 epochs, LR 2e-4) | This run (6 epochs, LR 3e-4) | Change |
|---|---:|---:|---|
| Cosine similarity (fine-tuned, same-cluster) | 0.8386 | 0.7571 | ↓ worse |
| LLM-judge composite (fine-tuned) | 3.89 | 3.73 | ↓ worse |
| Fine-tuning time | 3.7 min | 6.6 min | — |

### Full LLM-as-a-Judge (test split)
| Metric | Teacher | Baseline SLM | Fine-tuned SLM |
|---|---:|---:|---:|
| Equivalence | 5.00 | 3.35 | 3.45 |
| Faithfulness | 5.00 | 4.10 | 4.25 |
| Specificity | 5.00 | 3.40 | 3.50 |
| Composite | 5.00 | 3.62 | 3.73 |

## Conclusion
Increasing epochs and learning rate together made results worse, not better. Likely cause: with only 70 training examples, more epochs past a certain point causes the model to memorize training-specific quirks rather than learn generalizable patterns, and the higher learning rate compounded this drift. This shows up specifically on the held-out test set, which the model never saw during training.

## Next Steps
Isolate the two variables — test 6 epochs with the original learning rate (2e-4) alone to determine whether epochs or learning rate (or their combination) caused the regression. Also worth considering: increasing the training dataset size, since 70 examples is a small foundation for fine-tuning regardless of hyperparameters.
