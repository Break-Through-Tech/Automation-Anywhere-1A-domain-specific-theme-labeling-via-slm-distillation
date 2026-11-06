# Qwen2.5-3B — Domain-Specific Theme Labeling

**Break Through Tech AI | Automation Anywhere 1A**

An experimental comparison of baseline Qwen2.5-3B,
QLoRA fine-tuned Qwen2.5-3B, and Claude Haiku 4.5.

The objective is to study the trade-off between
**label quality, inference latency, and estimated cost**
for customer-support ticket cluster labeling.

## Research Question

Can QLoRA fine-tuning improve a small language model's
theme-labeling performance and narrow the gap to a
larger teacher model?

## Models

| Model | Role |
|---|---|
| Qwen2.5-3B-Instruct | Baseline SLM |
| Qwen2.5-3B-Instruct + QLoRA | Fine-tuned SLM |
| Claude Haiku 4.5 | Teacher LLM comparison |

## Experiment Setup

- Training examples: 70
- Validation examples: 15
- Held-out examples: 20
- Independent held-out ticket clusters: 4
- Prompt variants: 5 per cluster
- Predictions per model: 20
- Total pilot predictions: 60

The fine-tuned Qwen used 4-bit NF4 quantization,
LoRA rank 16, three epochs and a learning rate
of 2e-4.

The adapter was newly trained for this experiment;
the original historical adapter was not recovered.

## Results

| Metric | Qwen baseline | Qwen fine-tuned | Claude teacher |
|---|---:|---:|---:|
| Faithfulness | 3.70 | 4.10 | 4.40 |
| Relevance | 4.15 | 4.75 | 4.80 |
| Specificity | 3.00 | 3.65 | 3.80 |
| Composite / 5 | 3.6167 | 4.1667 | 4.3333 |
| Cosine similarity | 0.7252 | 0.8245 | 0.8870 |
| ROUGE-L F1 | 0.4036 | 0.5284 | 0.7127 |
| BERTScore F1 | 0.8969 | 0.9247 | 0.9512 |
| Mean latency (s) | 0.9843 | 1.3324 | 0.5044 |
| Est. USD / 1,000 | 0.2953 | 0.3997 | 0.3109 |
| 5-15 word compliance (%) | 50.0 | 100.0 | 100.0 |

## Key Findings

1. Fine-tuning increased Qwen's mean composite quality
   score from 3.62 to 4.17 out of 5.
2. Cosine similarity, ROUGE-L, and BERTScore improved
   after fine-tuning.
3. Fine-tuned Qwen had approximately 35% higher
   measured average generation latency.
4. Claude Haiku achieved the highest average
   label-quality scores in this pilot.
5. None of the cluster-level statistical comparisons
   reached the conventional p < 0.05 threshold.

## Evaluation Methodology

**Faithfulness, relevance and specificity**

All 60 outputs were judged using the same
Claude Sonnet 4.6 evaluation rubric.

Composite is the mean of these three scores.

**Cosine similarity**

`sentence-transformers/all-MiniLM-L6-v2`,
attention-masked mean pooling, L2 normalization.

**ROUGE-L**

ROUGE-L F1 with stemming enabled.

**BERTScore**

`roberta-large`, F1, without baseline rescaling.

**Statistics**

Paired cluster-level comparisons with exact
two-sided sign-flip tests and exploratory bootstrap
confidence intervals.

## Latency and Cost Methodology

Qwen results use GPU-synchronized generation-only
latency measurements from a Google Colab T4 GPU.

Claude results use full API round-trip time.

Qwen serving cost is an estimate based on a
hypothetical cloud GPU + VM price and 50% utilization.

Claude cost is calculated from reported API
input/output token usage and assumed token pricing.

**These latency and cost methods are not directly
equivalent. They should not be interpreted as
a production deployment benchmark.**

## Repository Structure

```text
experiments/qwen25_3b/
├── README.md
├── configs/
│   ├── training_config.json
│   └── cost_assumptions.json
├── notebooks/
│   └── qwen_pilot_analysis.ipynb
└── results/
    ├── three_model_summary.csv
    ├── cluster_statistics.csv
    └── anonymized_prediction_metrics.csv
```

## How to Explore the Results

Open `notebooks/qwen_pilot_analysis.ipynb`
in Google Colab or Jupyter Notebook.

Run its cells to display model comparisons,
quality metrics, and statistical tests.

This is a results-analysis notebook, not
a complete retraining pipeline.

## Data Availability and Privacy

The original 60-row, 17-column experiment CSV
is stored separately in the authorized project
Google Drive.

The public CSV here contains numerical metrics
and anonymous cluster labels only.

Original customer-support ticket text, generated
labels, model weights and API credentials are
not included in this contribution.

## Limitations and Future Work

- Only four independent ticket clusters.
- Five prompts for each cluster are correlated.
- Small-sample statistical uncertainty.
- Qwen runs were measured in different sessions.
- Teacher and SLM timing are not directly comparable.
- Inference cost estimates require deployment validation.
- The larger team-wide evaluation is still pending.

This is a completed **research pilot**, not the
full team-wide 5,500-row example described
in the challenge advisor's instructions.

## Contributor

**Daksh Kumar**

Qwen2.5-3B fine-tuning, benchmarking,
quality evaluation, statistical analysis and
three-model research comparison.
