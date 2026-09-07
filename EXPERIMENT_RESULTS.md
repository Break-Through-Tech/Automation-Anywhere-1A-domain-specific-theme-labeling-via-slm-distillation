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
