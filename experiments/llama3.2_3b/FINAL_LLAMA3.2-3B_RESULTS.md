# Llama-3.2-3B-Instruct Benchmark Results

## Model

- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Fine-tuning method: 4-bit QLoRA
- GPU: Tesla T4
- LoRA rank: 16
- LoRA alpha: 16
- Learning rate: 2e-4
- Training epochs: 3

## Dataset

- Total samples: 500
- Training epochs: 3
- Evaluation examples: 20
- Run ID: `20260914_0241_Llama-3.2-3B-Instruct_ep3`

## Training

Training completed successfully.

- Logged training loss: 2.1280
- Epoch 1 validation loss: 1.9152
- Epoch 2 validation loss: 1.4049
- Epoch 3 validation loss: 1.2854
- Best epoch: 3
- Best checkpoint: checkpoint-15

Validation loss improved at every epoch, with the best validation
performance occurring at epoch 3.

## Cosine Similarity

- Baseline similarity: 0.8211
- Fine-tuned similarity: 0.8199
- Change: -0.0012
- Baseline multi-reference similarity: 0.8795
- Fine-tuned multi-reference similarity: 0.8760

The cosine similarity decreased slightly after fine-tuning.

## ROUGE-L

- Baseline ROUGE-L: 0.5029
- Fine-tuned ROUGE-L: 0.5003
- Baseline multi-reference ROUGE-L: 0.6083
- Fine-tuned multi-reference ROUGE-L: 0.6318

The standard ROUGE-L score decreased slightly, while the
multi-reference ROUGE-L score improved after fine-tuning.

## BERTScore

- Baseline BERTScore F1: 0.9145
- Fine-tuned BERTScore F1: 0.9073
- Baseline multi-reference BERTScore F1: 0.9329
- Fine-tuned multi-reference BERTScore F1: 0.9293

BERTScore decreased slightly after fine-tuning.

## LLM-as-a-Judge

- Baseline equivalence: 3.45
- Fine-tuned equivalence: 3.70
- Baseline faithfulness: 4.35
- Fine-tuned faithfulness: 4.45
- Baseline specificity: 3.60
- Fine-tuned specificity: 3.70
- Baseline composite score: 3.80
- Fine-tuned composite score: 3.95

The fine-tuned model improved across all LLM-judge dimensions.

## Inference Performance

- Baseline inference latency: 0.913 seconds per label
- Fine-tuned inference latency: 0.767 seconds per label
- Baseline throughput: 65.74 labels/minute
- Fine-tuned throughput: 78.24 labels/minute
- Fine-tuning time: 2.13 minutes

The fine-tuned model produced labels faster than the baseline model.

## Conclusion

The Llama-3.2-3B-Instruct smoke test produced mixed results.

Fine-tuning slightly reduced cosine similarity and BERTScore, while
multi-reference ROUGE-L improved. The LLM-as-a-Judge evaluation also
improved, with the composite score increasing from 3.80 to 3.95.

Inference performance improved as well, with latency decreasing from
approximately 0.913 seconds per label to 0.767 seconds per label.

Overall, fine-tuning improved judged label quality and inference speed,
although several automatic similarity metrics decreased slightly.
