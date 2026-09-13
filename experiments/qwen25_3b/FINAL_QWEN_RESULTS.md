# Qwen2.5-3B-Instruct Benchmark Results

## Model

- Model: `Qwen/Qwen2.5-3B-Instruct`
- Fine-tuning method: 4-bit QLoRA
- GPU: Tesla T4
- LoRA rank: 16
- LoRA alpha: 16
- LoRA dropout: 0.05
- Learning rate: 2e-4
- Max sequence length: 384

## Dataset

- Training examples: 70
- Validation examples: 15
- Test examples: 20
- Prompt variants: P1-P5
- Training epochs: 3

## Training

Training completed successfully.

- Epoch 1 validation loss: 1.5322
- Epoch 2 validation loss: 0.9958
- Epoch 3 validation loss: 0.8789
- Best epoch: 3
- Best checkpoint: checkpoint-27

Validation loss improved across all three epochs.

## Cosine Similarity

- Baseline: 0.7252
- Fine-tuned: 0.8077
- Improvement: +0.0825
- Fine-tuned better on: 17/20
- Baseline better on: 3/20

## LLM-as-a-Judge

- Fine-tuned wins: 15
- Baseline wins: 5
- Ties: 0
- Baseline average score: 2.85
- Fine-tuned average score: 3.80

## Conclusion

Fine-tuning Qwen2.5-3B produced a clear improvement over the baseline.

The fine-tuned model achieved higher semantic similarity on most test
examples and was strongly preferred by the LLM judge.

Qwen2.5-3B therefore showed a positive response to the current QLoRA
configuration and is a strong candidate for the project.
