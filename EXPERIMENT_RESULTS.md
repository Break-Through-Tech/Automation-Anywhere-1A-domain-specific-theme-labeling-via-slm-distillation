# SmolLM2 SLM Distillation Experiment Results

## Model

- Student model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- Fine-tuning method: QLoRA / LoRA
- GPU: Tesla T4
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05

## Dataset

- Training examples: 70
- Validation examples: 15
- Test examples: 20
- Training epochs: 3
- Training steps: 15

## Training Result

- Training completed successfully
- Final training loss: 2.0383
- Fine-tuned adapter was saved to Google Drive

## Baseline vs Fine-Tuned Evaluation

### Cosine Similarity

| Model | Average Similarity |
|---|---:|
| Baseline SLM | 0.5246 |
| Fine-tuned SLM | 0.6004 |

Average improvement: **+0.0758**

The fine-tuned model performed better on **11 of 20** test examples.

### LLM-as-a-Judge

| Result | Count |
|---|---:|
| Fine-tuned wins | 9 |
| Baseline wins | 3 |
| Ties | 8 |

Average judge scores:

- Baseline SLM: **3.00**
- Fine-tuned SLM: **3.55**

## Conclusion

The fine-tuned SmolLM2 model improved overall theme-label quality compared with the baseline model. The largest improvements appeared on examples where the baseline produced overly generic labels. Some predictions still need improvement in formatting and instruction following, such as avoiding prefixes, quotation marks, or overly long responses.


<!-- MISTRAL7B_RESULTS_START -->

## Mistral-7B-Instruct-v0.3 Benchmark

### Setup

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Method: 4-bit QLoRA
- GPU: Tesla T4
- Train / validation / test: 70 / 15 / 20
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Max sequence length: 384
- Training epochs: 3
- Best checkpoint: epoch 2 (`checkpoint-18`)
- Best validation loss: 0.8091

### Evaluation Results

| Metric | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Cosine similarity | 0.7673 | 0.7318 |
| LLM judge average score | 3.40 | 3.25 |
| LLM judge wins | 10 | 10 |

Fine-tuned predictions had higher cosine similarity on **8/20**
test examples, while the baseline performed better on **12/20**.

### Conclusion

The baseline Mistral-7B-Instruct-v0.3 model performed slightly better
overall than the current QLoRA fine-tune.

Fine-tuning produced more detailed labels, but it did not improve the
overall semantic-similarity or LLM-judge metrics. Validation loss was
lowest at epoch 2 and increased at epoch 3, suggesting mild overfitting.

This is a useful negative benchmark result: the base Mistral model was
already strong on this task, and the current fine-tuning configuration
did not provide an overall improvement.

Detailed reproducibility files are available in:

`experiments/mistral7b/`

Notebook:

`notebooks/mistral7b_benchmark.ipynb`

Large adapters, optimizer states, and checkpoints are intentionally
stored outside GitHub.

<!-- MISTRAL7B_RESULTS_END -->

