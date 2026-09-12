# Mistral-7B-Instruct-v0.3 Benchmark Results

## Model

- Model: `mistralai/Mistral-7B-Instruct-v0.3`
- Fine-tuning method: 4-bit QLoRA
- GPU: Tesla T4
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Maximum sequence length: 384

## Dataset

- Training examples: 70
- Validation examples: 15
- Test examples: 20
- Prompt variants: P1, P2, P3, P4, P5
- Training epochs: 3

## Training

Training completed successfully.

- Final training loss: 0.8506
- Epoch 1 validation loss: 0.8444
- Epoch 2 validation loss: 0.8091
- Epoch 3 validation loss: 0.8962
- Best epoch: 2
- Best checkpoint: checkpoint-18

Validation performance improved through epoch 2 and declined at
epoch 3, suggesting mild overfitting.

## Cosine Similarity

- Baseline average similarity: 0.7673
- Fine-tuned average similarity: 0.7318
- Change: -0.0355
- Fine-tuned better on: 8/20
- Baseline better on: 12/20

## LLM-as-a-Judge

- Fine-tuned wins: 10
- Baseline wins: 10
- Ties: 0
- Baseline average score: 3.40
- Fine-tuned average score: 3.25

## Conclusion

The baseline Mistral-7B-Instruct-v0.3 model performed slightly better
overall than the fine-tuned model in this smoke test.

Fine-tuning produced more detailed and descriptive labels, but the
baseline achieved higher average cosine similarity and a slightly
higher LLM-judge score.

This suggests that the base Mistral model was already strong for this
task and that the current QLoRA configuration did not provide an
overall improvement.

Future experiments could test different learning rates, fewer epochs,
LoRA parameters, or response-only loss masking.
