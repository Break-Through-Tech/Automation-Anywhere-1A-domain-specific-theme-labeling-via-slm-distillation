# SLM Smoke Test Comparison

**Automation Anywhere | Break Through Tech AI Studio | Phase 1**  
**Compiled:** September 16, 2026  
**Evidence:** Team results supplied by Hari. Values are reported results, not independently rerun benchmarks.

## 1. Executive summary

Five of the eight requested models have completed results. Llama 3.1 8B and Gemma 2 9B have no results due to reported storage issues. Qwen3 8B was blocked by GPU memory exhaustion. SmolLM2-360M is included separately as a supplementary completed benchmark.

- **Qwen2.5 3B showed the largest reported cosine improvement** among the completed submissions: +0.0825. Its judge average rose from 2.85 to 3.80, with 15 of 20 judge comparisons favoring the fine-tuned model.
- **Gemma 3 4B improved on every reported quality measure**, but its fine-tuned inference latency was 1.781 seconds per label, compared with 0.80 seconds per teacher call in that report.
- **Llama 3.2 3B improved all judge dimensions and reduced latency by about 16%**, despite small decreases in most similarity measures.
- **Phi-3.5 Mini showed modest, mixed improvements:** cosine and BERTScore increased, but judge scores declined. Its reported fine-tuning time was 2.56 minutes.
- **Mistral 7B did not improve overall:** both cosine similarity and judge average decreased after fine-tuning.
- **Supplementary SmolLM2-360M improved both reported averages**, although formatting and instruction-following issues remained.

These are descriptive comparisons, not a controlled leaderboard. Test identities, reference aggregation, judge protocols, sequence lengths, and dataset reporting differ or remain unverified. No overall winner is established.

## 2. Model coverage and run status

| Requested model | Model identifier supplied | Status | Results availability |
|---|---|---|---|
| Llama 3.1 8B Instruct | Not supplied | Unavailable: reported storage issues | N/A; no detailed failure log supplied |
| Llama 3.2 3B Instruct | `meta-llama/Llama-3.2-3B-Instruct` | Completed | Quality, judge, latency, validation loss |
| Gemma 2 9B | Not supplied | Unavailable: reported storage issues | N/A; no detailed failure log supplied |
| Gemma 3 4B | `google/gemma-3-4b-it` | Completed after processor compatibility fix | Quality, judge, training time, latency, estimated cost |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | Completed | Cosine, judge, best validation loss |
| Phi-3.5 Mini Instruct | `microsoft/Phi-3.5-mini-instruct` | Completed; pipeline exit code 0 | Quality, judge, latency, throughput, training time, estimated cost |
| Qwen2.5 3B Instruct | `Qwen/Qwen2.5-3B-Instruct` | Completed | Cosine, judge, validation loss |
| Qwen3 8B | `Qwen/Qwen3-8B` | Blocked: CUDA out of memory | N/A; failed during LoRA adapter injection |

**Supplementary model:** `HuggingFaceTB/SmolLM2-360M-Instruct`, completed successfully.

**Notation:** NR means not reported in the supplied summary. N/A means no completed benchmark is available. Neither means zero performance. Storage issues and GPU VRAM exhaustion are distinct failure categories.

## 3. Experiment settings and comparability

| Completed model | Data reported | Epochs | Method | GPU | LoRA rank / alpha / dropout |
|---|---|---:|---|---|---|
| Phi-3.5 Mini | 500 raw tickets configured; split counts NR | 3 | 4-bit QLoRA | T4 | 16 / 16 / 0.05 |
| Llama 3.2 3B | 500 samples; split counts NR | 3 | 4-bit QLoRA | T4 | 16 / 16 / NR |
| Gemma 3 4B | 497 tickets; 21 clusters; 14 / 3 / 4 train / validation / test clusters | 3 | QLoRA / LoRA, as reported | T4 | 16 / 16 / 0.05 |
| Mistral 7B | 70 / 15 / 20 train / validation / test examples | 3 | 4-bit QLoRA | T4 | 16 / 16 / 0.05 |
| Qwen2.5 3B | 70 / 15 / 20 train / validation / test examples | 3 | 4-bit QLoRA | NR | 16 / 16 / 0.05 |
| SmolLM2-360M (supplementary) | 70 / 15 / 20 train / validation / test examples | 3 | QLoRA / LoRA, as reported | T4 | 16 / 16 / 0.05 |

Raw tickets, clusters, and training examples are different units. Do not interpret the 70 training examples as directly comparable to Phi's 500 configured tickets. Matching split counts also do not prove that models used the same examples.

| Setting | Phi-3.5 Mini | Mistral 7B | Qwen2.5 3B |
|---|---|---|---|
| Learning rate | 2e-4, saved configuration | 2e-4 | 2e-4 |
| Maximum sequence length | 2048, saved configuration | 384 | 384 |
| Per-device training batch | 1, saved configuration | NR | NR |
| Gradient accumulation | 16, saved configuration | NR | NR |

The corresponding settings were not supplied for the other completed models. Phi configuration values describe the intended run settings; implementation-level overrides were not inspected.

## 4. Headline quality comparison

Changes are **fine-tuned minus baseline**, in score units. Higher scores are better within each metric. Cosine and BERTScore values are not accuracy percentages.

| Model | Cosine baseline | Cosine fine-tuned | Change | Judge baseline | Judge fine-tuned | Change |
|---|---:|---:|---:|---:|---:|---:|
| Llama 3.1 8B | N/A | N/A | N/A | N/A | N/A | N/A |
| Llama 3.2 3B | 0.8211 | 0.8199 | -0.0012 | 3.80 | 3.95 | +0.15 |
| Gemma 2 9B | N/A | N/A | N/A | N/A | N/A | N/A |
| Gemma 3 4B | 0.8041 | 0.8386 | +0.0345 | 3.68 | 3.89 | +0.21 |
| Mistral 7B | 0.7673 | 0.7318 | -0.0355 | 3.40 | 3.25 | -0.15 |
| Phi-3.5 Mini | 0.793341 | 0.809278 | +0.015937 | 3.8500 | 3.7667 | -0.0833 |
| Qwen2.5 3B | 0.7252 | 0.8077 | +0.0825 | 2.85 | 3.80 | +0.95 |
| Qwen3 8B | N/A | N/A | N/A | N/A | N/A | N/A |
| SmolLM2-360M (supplementary) | 0.5246 | 0.6004 | +0.0758 | 3.00 | 3.55 | +0.55 |

**Metric definitions:** Phi uses same-prompt cosine and a three-dimension judge composite. Gemma labels its cosine measure “same-cluster.” Llama supplies cosine and a judge composite. Mistral, Qwen2.5, and SmolLM2 report average cosine and average judge scores without full aggregation details. The table preserves these reports without asserting that their protocols are identical. Gemma's source composite is retained as reported; its rounded dimension values do not exactly reproduce that composite.

## 5. Detailed similarity metrics

Each cell shows **baseline → fine-tuned**. Only models with these additional metrics supplied are included.

| Metric | Phi-3.5 Mini | Llama 3.2 3B | Gemma 3 4B |
|---|---|---|---|
| Primary cosine | 0.793341 → 0.809278 | 0.8211 → 0.8199 | 0.8041 → 0.8386 |
| Secondary cosine | 0.872364 → 0.891755 | 0.8795 → 0.8760 | 0.8679 → 0.8874 |
| Primary ROUGE-L | 0.541147 → 0.565275 | 0.5029 → 0.5003 | 0.5046 → 0.5354 |
| Secondary ROUGE-L | 0.713991 → 0.699896 | 0.6083 → 0.6318 | 0.6048 → 0.6393 |
| Primary BERTScore F1 | 0.924928 → 0.929018 | 0.9145 → 0.9073 | NR |
| Secondary BERTScore F1 | 0.948468 → 0.951208 | 0.9329 → 0.9293 | NR |

Phi labels these variants same-prompt / multi-reference; Llama reports primary / multi-reference; Gemma reports same-cluster / multi-cluster. “Primary” and “secondary” are presentation groupings only. In particular, do not assume Gemma's “multi-cluster” means cross-cluster comparison or is equivalent to multi-reference without checking its metric implementation.

## 6. Judge dimensions and example-level outcomes

| Judge dimension | Phi baseline → fine-tuned | Llama 3.2 baseline → fine-tuned | Gemma 3 baseline → fine-tuned |
|---|---|---|---|
| Equivalence | 3.50 → 3.40 | 3.45 → 3.70 | 3.30 → 3.55 |
| Faithfulness | 4.40 → 4.30 | 4.35 → 4.45 | 4.20 → 4.45 |
| Specificity | 3.65 → 3.60 | 3.60 → 3.70 | 3.55 → 3.65 |
| Composite | 3.8500 → 3.7667 | 3.80 → 3.95 | 3.68 → 3.89 |

| Model | Fine-tuned cosine wins | Baseline cosine wins | Judge fine-tuned wins | Judge baseline wins | Judge ties |
|---|---:|---:|---:|---:|---:|
| SmolLM2-360M | 11 / 20 | NR | 9 | 3 | 8 |
| Mistral 7B | 8 / 20 | 12 / 20 | 10 | 10 | Not separately reported |
| Qwen2.5 3B | 17 / 20 | 3 / 20 | 15 | 5 | 0 |

SmolLM2's remaining nine cosine comparisons were not broken down into losses versus ties. Gemma explicitly supplied aggregate scores only. No example-level win counts were supplied for Phi or Llama 3.2. Qwen2.5 used Claude Haiku 4.5 as judge; Phi's supplied configuration also selected Claude Haiku 4.5. Full judge settings were not supplied for every model.

Equal judge win counts for Mistral do not contradict a lower fine-tuned average: score magnitudes can differ between examples.

## 7. Training behavior

| Model | Validation loss by epoch | Selected checkpoint | Other training evidence |
|---|---|---|---|
| Llama 3.2 3B | 1.9152 → 1.4049 → 1.2854 | Epoch 3, `checkpoint-15` | Validation loss improved at each reported epoch |
| Qwen2.5 3B | 1.5322 → 0.9958 → 0.8789 | Epoch 3, `checkpoint-27` | Validation loss improved at each reported epoch |
| Mistral 7B | Full sequence NR; best loss 0.8091 at epoch 2 | Epoch 2, `checkpoint-18` | Reported increase at epoch 3 is consistent with possible overfitting, not proof |
| SmolLM2-360M | NR | NR | 15 training steps; final training loss 2.0383 |
| Phi-3.5 Mini | NR | Selected checkpoint NR | Saved checkpoints at steps 5, 10, and 15; adapter saved |
| Gemma 3 4B | NR | NR | Adapter saved; training completed |

Training loss and validation loss are different quantities. Loss values across different model tokenizers and sequence settings should not be used as a model leaderboard. Llama's falling validation loss alongside mixed test metrics also shows that lower training-objective loss need not improve every downstream measure.

## 8. Runtime, throughput, and costs

| Model | Training time | Baseline inference | Fine-tuned inference | Within-run latency change |
|---|---|---|---|---|
| Phi-3.5 Mini | 153.47 s / 2.56 min | 0.685575 s per cluster per prompt | 0.660370 s per cluster per prompt | Approximately 3.7% lower |
| Llama 3.2 3B | NR | 0.913 s per label | 0.767 s per label | Approximately 16.0% lower |
| Gemma 3 4B | 3.7 min | NR | 1.781 s per label | Cannot calculate without baseline latency |
| Mistral 7B | NR | NR | NR | NR |
| Qwen2.5 3B | NR | NR | NR | NR |
| SmolLM2-360M | NR | NR | NR | NR |

Phi throughput rose from **87.517779 to 90.858089 labels/minute**. Gemma model loading took **45.8 seconds for baseline** and **42.1 seconds for fine-tuned**; loading time is separate from inference latency.

| Report | Teacher labeling | Student cost reported | Cost qualification |
|---|---|---|---|
| Phi | 105 calls; estimated $0.01365 | $0 under free Colab assumption | Judge calls and total deployment costs not established |
| Gemma 3 | 105 calls; estimated $0.0137 | Approximately $0 under local/Colab assumption | Does not include a priced hardware or hosting comparison |
| Other models | NR | NR | No cost ranking possible |

Gemma reports **0.80 s per teacher call** versus **1.781 s per fine-tuned student label**. The raw ratio is about **2.23 times the latency**, not “0.5x faster.” Whether these operations represent equivalent workloads is unverified. Likewise, latency comparisons across models require matched input/output lengths, batching, warmups, and hardware conditions. One-run timing differences do not establish repeatable speed gains.

## 9. Engineering issues and unavailable runs

### Gemma 3 processor compatibility: resolved for the reported run

- Error: `AttributeError: 'Gemma3Processor' object has no attribute 'encode'` in `dataset.py`, around line 194.
- Team diagnosis: the pipeline received a processor rather than a plain tokenizer.
- Reported fix: call `.tokenizer.encode()` when the object has a `.tokenizer` attribute.
- The team reports that the patch was pushed to branch `btt_setup_mt`.
- This fix is relevant when other model loading paths return a processor; the supplied evidence does not establish that every Gemma-family model has the same issue.

### Qwen3 8B: blocked by GPU VRAM exhaustion

- Configuration: `Qwen/Qwen3-8B`, 4-bit QLoRA / LoRA, Tesla T4 with 14.56 GB reported VRAM.
- The pipeline loaded the model but failed during LoRA adapter injection with `torch.OutOfMemoryError`.
- The team suspects two full model instances were retained across dataset/tokenizer setup and fine-tuning. This is a reported hypothesis, not a verified root cause.
- Attempts: force `device_map={"": 0}`; reduce batch size from 4 to 1 and increase gradient accumulation; reduce sequence length from 2048 to 1024; restart the runtime.
- Forcing device placement reportedly resolved an earlier offload error, but the memory failure remained.
- Final reported failure: allocation of 2 MiB failed with 14.39 / 14.56 GB already used.
- Result: **N/A for completed training and evaluation**. This shows the current implementation did not fit; it does not establish that Qwen3 8B can never run on a T4 under a different implementation.

### Llama 3.1 8B and Gemma 2 9B: unavailable

Both are marked **N/A due to reported storage issues**, as requested by the team. No detailed logs, device measurements, or completed metric files were supplied. No more specific failure diagnosis is assigned.

### Phi adapter metadata: outstanding verification

The saved adapter has `base_model_name_or_path: null`. The configuration and training report identify Phi, and weights/checkpoints were saved, but reloading the adapter in a fresh session remains untested. This metadata gap alone does not demonstrate failed training.

## 10. Interpretation and next comparison steps

**Most encouraging fine-tuning response:** Qwen2.5 3B improved both headline measures and won most of its 20 example comparisons. SmolLM2 also benefited, while Gemma improved every supplied quality metric. These observations support further matched experiments, rather than a claim of statistical superiority.

**Mixed outcomes:** Llama 3.2 improved judged quality while most similarity metrics fell; Phi showed largely the opposite pattern. Teacher-label similarity and judged usefulness measure different aspects of output quality.

**Negative result worth retaining:** Mistral's baseline outperformed its fine-tune on both average cosine and judge score. Its strong baseline and short fine-tuning experiment provide useful evidence that adaptation is not automatically beneficial.

Before selecting a final model:

1. Use identical held-out clusters, teacher labels, prompt variants, reference aggregation, and judge rubrics across candidates. Preserve cluster-level splits to avoid leakage.
2. Record exact run settings and processed example counts, including token limits and GPU type. A configured ticket limit does not prove the final processed count.
3. Preserve per-example outputs and repeat runs before treating small changes as reliable. Prompt variants from the same cluster should not be treated as independent tickets.
4. Report teacher reference scores separately from independent teacher evaluation. Phi explicitly uses hardcoded perfect teacher judge scores; perfect teacher values elsewhere do not prove independently measured superiority.
5. Collect matched latency, throughput, and complete cost measurements for models with missing business metrics.
6. Investigate Qwen3 model lifetime and memory allocation before retrying; diagnose the separate storage failures from their logs.
7. Verify adapter reloads and retain model identifiers, configurations, and exact code versions with each run.

## 11. Provenance and reproducibility

This report compiles the summaries pasted into the conversation. Linked repository files were not independently inspected for this comparison. NR fields remain missing rather than being inferred from other teammates' runs.

| Model | Supplied run or artifact reference |
|---|---|
| Phi-3.5 Mini | Run `20260913_2207_Phi-3.5-mini-instruct_ep3`; branch `btt_setup_hn`; initial artifact commit `7e72b28` |
| Llama 3.2 3B | Run `20260914_0241_Llama-3.2-3B-Instruct_ep3` |
| Gemma 3 4B | Run `20260915_0252_gemma-3-4b-it_ep3`; branch `btt_setup_mt` |
| Mistral 7B | Reported paths: `experiments/mistral7b/` and `notebooks/mistral7b_benchmark.ipynb`; branch NR |
| Qwen2.5 3B | Best checkpoint `checkpoint-27`; run ID and artifact path NR |
| SmolLM2-360M | Adapter reported saved to Drive; run ID and artifact path NR |
| Qwen3 8B | Failure report in `experiment-results.md` on `btt_setup_mt` |

- [Phi notebook](https://github.com/Break-Through-Tech/Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation/blob/btt_setup_hn/notebooks/Hari_Phi_Smoke_Test.ipynb)
- [Phi committed configuration and metric summaries](https://github.com/Break-Through-Tech/Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation/commit/7e72b28)
- [Team Gemma 3 and Qwen3 report](https://github.com/Break-Through-Tech/Automation-Anywhere-1A-domain-specific-theme-labeling-via-slm-distillation/blob/btt_setup_mt/experiment-results.md)

**Meeting takeaway:** The completed smoke tests show that the pipeline can fine-tune and evaluate several SLMs, but gains depend on the model and metric. Qwen2.5, Gemma 3, and supplementary SmolLM2 show encouraging quality gains; Llama and Phi are mixed; Mistral regressed. Three requested models lack completed results because of reported storage or GPU-memory constraints. A matched evaluation is needed before choosing a winner.
