# Team SLM Fine-Tuning Results Summary

This file compiles the experimental results, debugging notes, and hyperparameter tests across various Small Language Models (SLMs) evaluated by the team for ticket distillation and classification tasks.

---

## 1. Gemma 3 4B — Initial Smoke Test (Sept 15, 2026)

* **Status:** Successful
* **Student Model:** `google/gemma-3-4b-it`
* **Setup:** QLoRA / LoRA (rank 16, alpha 16, dropout 0.05), Tesla T4, 3 epochs.
* **Dataset:** 497 tickets, 21 clusters (Train: 14, Val: 3, Test: 4).

### Evaluation Results (Test Split)
* **Cosine Similarity (Same-cluster):** Baseline 0.8041 → Fine-tuned **0.8386** (+0.0345)
* **ROUGE-L (Same-cluster):** Baseline 0.5046 → Fine-tuned **0.5354**
* **LLM-as-a-Judge Composite:** Baseline 3.68 → Fine-tuned **3.89** (Faithfulness: 4.45, Specificity: 3.65, Equivalence: 3.55)
* **Business Evaluation:** Mean latency 1.781s/label (fine-tuned); estimated cost ~$0.00 local/Colab.

### Notes & Issues
* **Bug Hit:** `AttributeError: 'Gemma3Processor' object has no attribute 'encode'` in `dataset.py`. Gemma 3 initializes with a `Processor` object instead of a plain tokenizer.
* **Fix:** Patched locally by calling `.tokenizer.encode()` when a `.tokenizer` attribute is present; pushed to branch `btt_setup_mt`[cite: 1].

---

## 2. Qwen3 8B — Initial Smoke Test & Resolution (Sept 15 & Sept 29, 2026)

* **Status:** Resolved (Previously Blocked: Out of Memory)
* **Student Model:** `Qwen/Qwen3-8B`
* **Setup:** QLoRA / LoRA 4-bit (rank 16, alpha 16, dropout 0.05), Tesla T4 (14.56 GB VRAM), 3 epochs.

### Root Cause & Code Fix (Memory Bug)
* **The Issue:** `pipeline.py` loaded a temporary model during dataset construction to build the tokenizer, but discarded it without freeing GPU memory. When the actual training model loaded afterward, both copies competed for VRAM, triggering a `torch.OutOfMemoryError` on the T4.
* **The Fix:** Added an explicit `del temp_model, tokenizer_tmp` followed by `_clear_device_cache()` before the real model load. This cleanup logic prevents multi-model VRAM contention and is essential for 7B–9B class models on T4 GPUs[cite: 1].

### Evaluation Results (Test Split — Post-Fix)
* **Cosine Similarity (Same-cluster):** Baseline 0.4333 → Fine-tuned **0.3406** (-0.0927) *(shift away from teacher phrasing)*
* **ROUGE-L (Same-cluster):** Baseline 0.1136 → Fine-tuned **0.1073**
* **LLM-as-a-Judge Composite:** Baseline 2.92 → Fine-tuned **3.95** (Equivalence: 3.95, Faithfulness: 4.30, Specificity: 3.60)
* **Business Evaluation:** Mean latency 5.434s/label; 0.1x teacher speed.

---

## 3. Gemma 3 4B — Hyperparameter Experiment (Sept 29, 2026)

* **Status:** Experiment Complete — Regression Observed
* **Student Model:** `google/gemma-3-4b-it`
* **Setup:** Increased epochs (3 → 6) and learning rate (2e-4 → 3e-4) simultaneously.

### Results vs. Original 3-Epoch Run
* **Cosine Similarity (Same-cluster):** 0.8386 (3 ep) → **0.7571** (6 ep) — *Worse*
* **LLM-Judge Composite:** 3.89 (3 ep) → **3.73** (6 ep) — *Worse*

### Conclusion
* Increasing epochs and learning rate together led to overfitting on the small training distribution (70 examples), harming generalization on the held-out test set[cite: 1].

---

## 4. Phi-3.5 Mini Optimization: Experiment A (Sept 29, 2026)

* **Status:** Target Exceeded (Validation Composite > 4.0)
* **Student Model:** `microsoft/Phi-3.5-mini-instruct`
* **Setup:** 4-bit QLoRA via Hugging Face/PEFT, FP16 compute on Tesla T4, 3 epochs, LR 0.0002, rank/alpha 16. Run ID: `20260929_2327_Phi-3.5-mini-instruct_ep3`[cite: 1].

### Validation Results (15 predictions across 3 validation clusters)
* **Judge Composite / 5:** Baseline 4.0000 → Fine-tuned **4.0444** (Target met)
* **Equivalence / 5:** Baseline 3.6667 → Fine-tuned **3.8000**
* **Faithfulness / 5:** Baseline 4.5333 → Fine-tuned **4.5333** (Unchanged)
* **Specificity / 5:** Baseline 3.8000 → Fine-tuned **3.8000** (Unchanged)
* **Multi-Reference Cosine Similarity:** Baseline 0.8354 → Fine-tuned **0.8714**
* **Inference Latency:** Baseline 1.616s → Fine-tuned **1.982s**

### Key Pipeline Updates Implemented
* Replaced full model loading during dataset construction with tokenizer-only loading.
* Released baseline and fine-tuned model references explicitly between evaluation stages.
* Handled quantization preparation using `prepare_model_for_kbit_training` and integrated `SFTConfig`[cite: 1].

---

## 5. Llama-3.2-3B-Instruct — 5-Epoch Run (Sept 29, 2026)

* **Status:** Successful — Clear Quantitative Gains
* **Student Model:** `meta-llama/Llama-3.2-3B-Instruct`
* **Setup:** LoRA fine-tuning, 5 epochs. Run ID: `20260929_2353_Llama-3.2-3B-Instruct_ep5`[cite: 1].

### Evaluation Results (Test Split)
* **Cosine Similarity (Same / Multi):** Fine-tuned **0.842 / 0.899** (Δ +0.036 / +0.051 over baseline)
* **ROUGE-L (Same / Multi):** Fine-tuned **0.559 / 0.731** (Δ +0.058 / +0.156 over baseline)
* **BERTScore F1 (Same / Multi):** Fine-tuned **0.922 / 0.943** (Δ +0.009 / +0.016 over baseline)
* **LLM-as-a-Judge Composite:** Baseline 3.63 → Fine-tuned **3.98** (Equivalence: 3.85, Faithfulness: 4.50, Specificity: 3.60)

### Conclusion
* Moving from 3 to 5 epochs turned LoRA fine-tuning into a consistent improvement across all embedding and lexical metrics, helping the student match the teacher's exact phrasing and structure[cite: 1].

---

## 6. Llama 3.1 8B — Post-Training Evaluation Memory Bug

* **Status:** In Progress / Pending Final Evaluation Run
* **Student Model:** `meta-llama/Llama-3.1-8B`
* **Setup:** 4-bit QLoRA, rank 16, alpha 16, dropout 0.05, batch size 1, grad accumulation 16, 3 epochs. Target modules explicitly configured to `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` to bypass 'all-linear' target errors[cite: 1].

### Current Obstacle & Next Steps
* Training completed successfully (loss dropped from 1.536 to 0.9515, and the LoRA adapter was saved to disk).
* **The Issue:** During the evaluation stage, the pipeline attempts to load Llama 3.1 8B again, causing a VRAM overflow on the Tesla T4 because the prior instance is not purged.
* **Next Steps:** Apply the same memory cleanup pattern used in the Qwen3-8B fix—explicitly deleting model references and clearing the CUDA cache prior to starting the evaluation stage—then load from the saved adapter to capture final metrics[cite: 1].
