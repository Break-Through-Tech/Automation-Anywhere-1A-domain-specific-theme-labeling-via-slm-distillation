# Llama 3.2-3B Phase 1 Smoke Test Results

## Model
meta-llama/Llama-3.2-3B-Instruct

## Setup
- Samples: 500
- Epochs: 3
- Environment: Google Colab
- Fine-tuning: QLoRA
- Run ID: 20260914_0241_Llama-3.2-3B-Instruct_ep3

## Results

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Cosine similarity | 0.8211 | 0.8199 |
| Cosine similarity (multi-ref) | 0.8795 | 0.8760 |
| ROUGE-L | 0.5029 | 0.5003 |
| ROUGE-L (multi-ref) | 0.6083 | 0.6318 |
| BERTScore F1 | 0.9145 | 0.9073 |
| BERTScore F1 (multi-ref) | 0.9329 | 0.9293 |
| LLM judge composite | 3.80 | 3.95 |
| Inference latency (sec/label) | 0.913 | 0.767 |

## LLM Judge

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Equivalence | 3.45 | 3.70 |
| Faithfulness | 4.35 | 4.45 |
| Specificity | 3.60 | 3.70 |
| Composite | 3.80 | 3.95 |

## Summary

Fine-tuning produced mixed metric results. Most cosine and BERTScore
metrics decreased slightly, while multi-reference ROUGE-L improved.

The fine-tuned model improved on all LLM judge dimensions and reduced
inference latency from approximately 0.913 seconds per label to 0.767
seconds per label.

Overall, the Llama 3.2-3B Phase 1 smoke test completed successfully.
