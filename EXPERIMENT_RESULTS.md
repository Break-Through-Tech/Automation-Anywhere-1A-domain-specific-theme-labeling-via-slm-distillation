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

<!-- QWEN25_3B_RESULTS_START -->

## Qwen2.5-3B-Instruct

**Model:** `Qwen/Qwen2.5-3B-Instruct`

**Method:** 4-bit QLoRA

### Training

- Training examples: 70
- Validation examples: 15
- Test examples: 20
- Epochs: 3
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Max sequence length: 384
- Best epoch: 3
- Best checkpoint: `checkpoint-27`
- Best validation loss: 0.8789

Validation loss improved across all three epochs:

- Epoch 1: 1.5322
- Epoch 2: 0.9958
- Epoch 3: 0.8789

### Cosine Similarity

- Baseline: **0.7252**
- Fine-tuned: **0.8077**
- Improvement: **+0.0825**
- Fine-tuned better: **17/20**
- Baseline better: **3/20**

### LLM-as-a-Judge

Using Claude Haiku 4.5:

- Fine-tuned wins: **15**
- Baseline wins: **5**
- Ties: **0**
- Baseline average score: **2.85**
- Fine-tuned average score: **3.80**

### Conclusion

Qwen2.5-3B showed a clear positive response to QLoRA fine-tuning.
The fine-tuned model improved semantic similarity from 0.7252 to
0.8077 and was preferred by the LLM judge on 15 of 20 test examples.

<!-- QWEN25_3B_RESULTS_END -->
