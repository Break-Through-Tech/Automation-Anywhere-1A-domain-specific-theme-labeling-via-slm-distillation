# Team SLM Fine-Tuning Results Summary

This file compiles experimental results across Small Language Models (SLMs).

## 1. Gemma 3 4B — Initial Smoke Test
- Status: Successful
- Cosine Similarity: 0.8386

## 2. Qwen3 8B — OOM Fix
- Status: Resolved via memory cleanup (`del temp_model`)

## 3. Phi-3.5 Mini Optimization
- Status: Target exceeded (4.0444/5 composite)

## 4. Llama-3.2-3B-Instruct
- Status: Successful (5 epochs, ROUGE-L multi +0.156)
