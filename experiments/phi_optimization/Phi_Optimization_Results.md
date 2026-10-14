# Phi-3.5 Mini Optimization and Repeated Inference Results

**Author:** Harinandan Nisheed (Hari)  
**Project:** Break Through Tech AI Studio, Automation Anywhere  
**Date:** October 13, 2026  
**Model:** `microsoft/Phi-3.5-mini-instruct`  
**Environment:** Google Colab, NVIDIA Tesla T4  
**GitHub branch:** `btt_setup_hn`

## Summary

Experiment B completed successfully and achieved a reported validation LLM-judge composite of **4.1111/5**, compared with **4.0222/5** for its baseline and **4.0444/5** for the earlier fine-tuned Experiment A. Most similarity metrics improved over B's baseline, while same-reference cosine similarity decreased slightly. The desired 4.2–4.4 composite has not yet been reached.

A separate repeated inference benchmark measured teacher, baseline, and fine-tuned latency on identical inputs. Experiment B's average latencies were **0.6342, 1.5878, and 2.0690 seconds per cluster label**, respectively. The fine-tuned student currently has a latency disadvantage. The experiment requesting 1,000 tickets also completed successfully. Its fine-tuned validation composite was 3.7333/5 versus 3.6533/5 for its own baseline. Its validation set changed after reclustering, so this is not a direct comparison with Experiment B's 4.1111/5.

## Dataset and training configuration

The existing teacher-labeled dataset was reused without repeating clustering or teacher label generation. It contained 497 tickets in 21 clusters. Training examples represent cluster-and-prompt combinations, rather than individual tickets.

| Setting | Experiment A | Experiment B |
|---|---|---|
| Configured ticket count | 500 | 500 |
| Actual tickets / clusters | 497 / 21 | 497 / 21 |
| Train / validation / test clusters | 14 / 3 / 4 | 14 / 3 / 4 |
| Train / validation / test examples | 70 / 15 / 20 | 70 / 15 / 20 |
| Fine-tuning method | 4-bit QLoRA | 4-bit QLoRA |
| Learning rate | 0.0002 | **0.0001** |
| Epochs | 3 | 3 |
| LoRA rank / alpha / dropout | 16 / 16 / 0.05 | 16 / 16 / 0.05 |
| Training batch size / gradient accumulation | 1 / 16 | 1 / 16 |
| Maximum training sequence length | 2048 | 2048 |
| Gradient checkpointing | Enabled | Enabled |
| Evaluation split for quality | Validation | Validation |

The lower learning rate was selected as a controlled hyperparameter experiment. It was not a guarantee of a higher score. A fresh adapter was trained from the base model, rather than continuing training Experiment A's adapter.

## Code changes that enabled training

### Existing memory and trainer compatibility fixes

The following changes from the earlier optimization work were retained:

- In `code/phase1/pipeline.py`, dataset construction loads only the tokenizer, avoiding a temporary full model before training.
- Baseline and fine-tuned evaluation release model references between stages to reduce simultaneous GPU memory use.
- In `code/phase1/finetuning/trainer.py`, the Colab loader returns a clean quantized base model. Adapters are attached separately.
- Quantized training uses `prepare_model_for_kbit_training`, and the trainer integrates `SFTConfig` with the supported `max_length` and `processing_class` interfaces.
- Adapter metadata records its base model identifier when saved.
- An explicit evaluation split selector allows validation inference during optimization, and BERTScore is enabled alongside cosine similarity, ROUGE-L, and LLM judging.

### New mixed-precision fix

The first Experiment B attempt failed before completing its first training step with:

```text
NotImplementedError: "_amp_foreach_non_finite_check_and_unscale_cuda"
not implemented for 'BFloat16'
```

The FP16 gradient scaler encountered BFloat16 gradients. The trainer was patched immediately before `trainer.train()` to convert only trainable parameters to FP32 and log their dtypes:

```python
for parameter in trainer.model.parameters():
    if parameter.requires_grad:
        parameter.data = parameter.data.to(torch.float32)

trainable_dtypes = {
    str(p.dtype) for p in trainer.model.parameters()
    if p.requires_grad
}
assert trainable_dtypes == {"torch.float32"}, trainable_dtypes
```

This preserves the quantized base model while keeping trainable adapter parameters in FP32. The source was backed up and checked for Python syntax before retrying. The retry completed with **pipeline exit code 0**.

The current environment reported Transformers 5.17.0 and rejected `warmup_ratio`, which the compatibility code removed. Package versions and effective warm-up behavior differ from the earlier run, so the A-to-B change cannot be attributed exclusively to learning rate.

## Experiment B validation results

Quality evaluation used **15 predictions across three validation clusters**, covering prompts P1–P5. Similarity scores are not accuracy percentages.

| Metric | Baseline B | Fine-tuned B |
|---|---:|---:|
| Judge equivalence / 5 | 3.6667 | 3.8000 |
| Judge faithfulness / 5 | 4.6000 | 4.6000 |
| Judge specificity / 5 | 3.8000 | 3.9333 |
| Judge composite / 5 | 4.0222 | **4.1111** |
| Cosine similarity, same reference | 0.7727 | 0.7694 |
| Cosine similarity, multi-reference | 0.8354 | 0.8627 |
| ROUGE-L, same reference | 0.5029 | 0.5215 |
| ROUGE-L, multi-reference | 0.6599 | 0.6794 |
| BERTScore F1, same reference | 0.9163 | 0.9169 |
| BERTScore F1, multi-reference | 0.9370 | 0.9420 |

Fine-tuning improved equivalence and specificity over baseline; faithfulness was unchanged. Its composite increased by **0.0889 points** over B's baseline and by **0.0667 points** over the earlier fine-tuned A result.

Fine-tuning took approximately **5.1 minutes**. The complete successful pipeline took **24.52 minutes**, including loading and evaluation. Pipeline validation inference averaged 1.722 seconds for baseline and 2.010 seconds for fine-tuned Phi. These are separate from the repeated benchmark below.

## How all three latency measurements were obtained

The original interactive demo loads two student models together and includes scoring calls. For the latency benchmark, a separate Colab cell called the same prompt-building and label-generation functions while loading only one student model at a time.

1. Recover the original held-out test cluster IDs and select their top-ranked tickets.
2. Use the same four clusters, top-k ticket selection, and **P1** prompt for every model.
3. Call the teacher API through `_get_llm_caller` and `build_messages`.
4. Load baseline Phi once, call `build_inference_prompt` and `generate_label`, then release it.
5. Load Phi with the saved Experiment B adapter using `PeftModel.from_pretrained`, generate labels, then release it.
6. Run three warm-up requests per model, excluded from measurements. Repeat every cluster five times, producing **20 measured requests per model**.
7. Measure with `time.perf_counter()`. Synchronize CUDA before and after student generation so timing captures completed GPU work.
8. Save every label and timing to `latency_raw.csv`; aggregate mean, median, standard deviation, and 95th percentile into `latency_summary.csv`. Save model loading times separately.

The student timings include tokenization, generation, and decoding; prompt construction, model loading, warm-ups, and quality scoring are excluded. Student decoding remains greedy with a 60-new-token cap. Teacher timing includes API/network overhead and the temperature-parameter fallback, including any retries within the caller. Teacher sampling settings were not identical to greedy student decoding.

### Experiment B repeated benchmark

| Model | Requests | Mean seconds | Median seconds | Standard deviation | 95th percentile | Mean output words |
|---|---:|---:|---:|---:|---:|---:|
| Teacher LLM | 20 | **0.6342** | 0.6267 | 0.0495 | 0.7058 | 7.20 |
| Baseline Phi | 20 | **1.5878** | 1.5931 | 0.0917 | 1.7024 | 5.25 |
| Fine-tuned Phi B | 20 | **2.0690** | 2.0875 | 0.2566 | 2.3485 | 6.25 |

Fine-tuned B was approximately **30.3% slower than baseline** and took **3.26 times the teacher's latency**. Longer outputs and adapter computation are possible contributors, but these measurements do not isolate their effects.

### Comparison with the earlier adapter benchmark

| Measurement | Experiment A adapter benchmark | Experiment B adapter benchmark |
|---|---:|---:|
| Teacher mean seconds | 0.6630 | 0.6342 |
| Baseline mean seconds | 1.4182 | 1.5878 |
| Fine-tuned mean seconds | 1.7430 | 2.0690 |

Baseline latency also increased between sessions, despite identical example baseline labels. This limits causal interpretation of the fine-tuned slowdown. Quality scores above come from validation, while the repeated latency benchmark uses four test clusters and only P1; they describe different workloads.

## Interpretation and limitations

- The reported quality gain is modest and based on a small validation set. Repeated judging and a final held-out quality evaluation are needed to establish reliability.
- Teacher judge scores of 5/5 are hardcoded in reference mode and are not an independent teacher-quality measurement.
- The judge caller removed the rejected temperature parameter. Deterministic judging is therefore not established.
- No improvement in both quality and speed has been demonstrated yet. The teacher API was fastest in these measurements.
- The training pipeline reported teacher latency and cost as zero because teacher labels were reused and no labeling latency records were available. These are missing-data placeholders, not measured zero latency or total cost. Repeated inference and judge calls incur separate API charges.
- Two training runs do not establish a relationship between fine-tuning duration and inference latency. More epochs increase training work without adding generation steps; comparable repeated experiments are needed to study any relationship.

## Larger dataset experiment: completed

**Run ID:** `20261014_0008_Phi-3.5-mini-instruct_ep3`  
**Status:** completed, pipeline exit code 0, with no out-of-memory failure in the supplied log.

The configuration requested **1,000 tickets** using `dataset.n_samples = 1000`, preserving Experiment B's learning rate of 0.0001, three epochs, rank 16, alpha 16, and dropout 0.05. Fresh data and checkpoint directories prevented reuse of the 500-ticket clustering results. Clustering, preprocessing, and teacher labeling were rerun; the quantization, sequential model cleanup, gradient checkpointing, and FP32 trainable-weight fix remained in place.

The supplied final log evaluated **25 validation predictions**. It did not confirm the actual retained ticket count or the training example count; 1,000 is the requested count, not a verified count of training examples.

### Larger-run validation quality

| Metric | Baseline | Fine-tuned |
|---|---:|---:|
| Judge composite / 5 | 3.6533 | **3.7333** |
| Equivalence / 5 | 3.2800 | 3.4000 |
| Faithfulness / 5 | 4.2400 | 4.2800 |
| Specificity / 5 | 3.4400 | 3.5200 |
| Cosine similarity, same reference | 0.7248 | 0.7961 |
| Cosine similarity, multi-reference | 0.7783 | 0.8302 |
| ROUGE-L, same reference | 0.4711 | 0.4752 |
| ROUGE-L, multi-reference | 0.5540 | 0.5650 |
| BERTScore F1, same reference | 0.8979 | 0.9094 |
| BERTScore F1, multi-reference | 0.9108 | 0.9217 |

Every reported quality metric improved over the larger run's own baseline. Its judge composite improved by **0.0800 points**, but did not reach the 4.2 target.

### Larger-run pipeline latency and training time

| Measurement | Reported result | Measurement source |
|---|---:|---|
| Teacher mean latency | 0.658 seconds | 190 teacher labeling calls |
| Baseline mean latency | 1.807 seconds | 25 validation predictions |
| Fine-tuned mean latency | 2.061 seconds | 25 validation predictions |
| Fine-tuning time | 9.2 minutes | Pipeline training timer |
| Full pipeline time | 43.71 minutes | Successful run wall time |
| Estimated teacher labeling cost | $0.0247 | Pipeline estimate; excludes unaccounted judge costs |

Fine-tuned validation inference was approximately **14.1% slower than baseline**. Teacher and student latency figures cover different workloads and request counts. These are pipeline measurements, not a repeated benchmark on identical teacher/student requests. The pipeline's phrase "0.3x faster" should not be interpreted as a speed advantage: the reported student latency is higher than teacher latency.

### What this experiment shows

Compared with Experiment B, training time increased from approximately **5.1 to 9.2 minutes**, while pipeline fine-tuned inference latency remained near two seconds (2.010 versus 2.061 seconds). This is descriptive evidence from two different workloads, not proof of a relationship between training time and inference speed.

The new validation baseline also scored lower (3.6533 versus B's 4.0222), and validation predictions increased from 15 to 25. Reclustering changed the evaluation inputs, so the lower absolute fine-tuned score does not establish that adding data harmed quality. Both adapters should be evaluated on the same genuinely held-out inputs before selecting a winner. Original benchmark inputs may overlap with training in the larger run and must be checked before using them for held-out quality claims.

**Current conclusion:** fine-tuning improved quality against each run's baseline, but neither a reliable 4.2 composite nor a student latency advantage has been demonstrated. No further adapter training is needed to preserve and share these completed results.

### Larger-run files

Configuration: `code/configs/phi_1000_lr1e4.yaml`

```text
/content/drive/MyDrive/slm-distillation/phi_optimization/
phi_1000_lr1e4_20261014_000827/outputs/
20261014_0008_Phi-3.5-mini-instruct_ep3/
```

This directory contains the saved adapter and `evaluation/metrics_summary.csv`, `evaluation/judge_summary.csv`, and `evaluation/business_eval.csv`. The parent experiment folder contains `pipeline.log`. A separate repeated three-model benchmark has not yet been reported for this adapter.

## Reproducibility files and locations

- Training configuration: `code/configs/phi_optimization_B.yaml`
- Patched source: `code/phase1/pipeline.py` and `code/phase1/finetuning/trainer.py`
- Successful B run: `20261013_2334_Phi-3.5-mini-instruct_ep3`
- B Drive run directory:

```text
/content/drive/MyDrive/slm-distillation/phi_optimization/
experiment_B_lr1e4_20261013_232940/outputs/
20261013_2334_Phi-3.5-mini-instruct_ep3/
```

Within that run directory:

```text
models/lora_adapter/
evaluation/judge_summary.csv
evaluation/metrics_summary.csv
evaluation/business_eval.csv
latency_benchmark_20261014_000250/latency_raw.csv
latency_benchmark_20261014_000250/latency_summary.csv
latency_benchmark_20261014_000250/model_loading.csv
latency_benchmark_20261014_000250/inputs.json
latency_benchmark_20261014_000250/inference_config.yaml
```

The successful training log is in the parent experiment folder as `pipeline.log`. Results are saved on Drive; this report does not confirm a GitHub commit or push of Experiment B's code and files.
