"""
phase1/evaluation/metrics.py

Shared evaluation metrics for SLM-generated cluster labels.

All metric functions support multi-reference evaluation:
given a generated label and multiple reference labels,
the score is the MAXIMUM over all references.

This is the shared module that must NOT be modified by students.
All 5 pairs import and use this module to ensure cross-pair comparability.
"""

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Public: run full evaluation ───────────────────────────────────────────────

def run_evaluation(
    cfg: dict,
    labeled_df: pd.DataFrame,
    predictions_path: str,
    fine_tuned: bool,
    output_path: str,
    eval_label: str = None,
    tag: str = None,               # "teacher" | "baseline" | "finetuned" — drives nonllm_{tag}.csv
) -> pd.DataFrame:
    """
    Evaluate a set of predictions against teacher labels.

    Saves two files:
      - output_path            : full per-prediction results with individual metric columns (internal use)
      - nonllm_{tag}.csv       : one row per (cluster, prompt) with a JSON metrics blob
                                 (portable, combine-friendly format)
    """
    import json
    from pathlib import Path
    from phase1.data.schema import (
        CLUSTER_ID, PROMPT_IDS, cluster_name_col,
        PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_GENERATED_LABEL, PRED_MODEL,
        EVAL_COSINE_SAME_PROMPT, EVAL_COSINE_MULTI_REF,
        EVAL_BERTSCORE_SAME, EVAL_BERTSCORE_MULTI,
        EVAL_ROUGEL_SAME, EVAL_ROUGEL_MULTI,
        nonllm_file,
    )

    eval_cfg  = cfg["evaluation"]
    model_id  = cfg["teacher_llm"]["model"]

    preds = []
    with open(predictions_path, "r") as f:
        for line in f:
            preds.append(json.loads(line))
    logger.info(f"[metrics] Evaluating {len(preds)} predictions ...")

    teacher_labels = _build_teacher_lookup(labeled_df, model_id)
    embed_model    = _load_embed_model(eval_cfg["embedding_model"])
    rouge_scorer   = _load_rouge()

    results = []
    for pred in preds:
        cid   = pred[PRED_CLUSTER_ID]
        pid   = pred[PRED_PROMPT_ID]
        gen   = pred[PRED_GENERATED_LABEL]
        model = pred.get(PRED_MODEL, tag or "unknown")

        if cid not in teacher_labels:
            logger.warning(f"[metrics] Cluster {cid} not found in teacher labels. Skipping.")
            continue

        cluster_refs = teacher_labels[cid]
        same_ref     = cluster_refs.get(pid, "")
        all_refs     = [v for v in cluster_refs.values() if v]

        cos_same  = cosine_similarity(gen, same_ref, embed_model)
        cos_multi = max_over_refs(gen, all_refs, cosine_similarity, embed_model)
        rl_same   = rouge_l(gen, same_ref, rouge_scorer)
        rl_multi  = max_over_refs(gen, all_refs, rouge_l, rouge_scorer)

        row = {
            PRED_CLUSTER_ID:        cid,
            PRED_PROMPT_ID:         pid,
            "model":                model,
            "fine_tuned":           fine_tuned,
            PRED_GENERATED_LABEL:   gen,
            EVAL_COSINE_SAME_PROMPT: cos_same,
            EVAL_COSINE_MULTI_REF:   cos_multi,
            EVAL_ROUGEL_SAME:        rl_same,
            EVAL_ROUGEL_MULTI:       rl_multi,
        }
        results.append(row)

    results_df = pd.DataFrame(results)

    run_bertscore = eval_cfg.get("run_bertscore", False)
    results_df    = _add_bertscore(
        results_df, eval_cfg["bertscore_model"], labeled_df, model_id, run_bertscore
    )

    # ── Save full results (internal, in _cache/) ──────────────────────────────
    cache_dir = Path(output_path).parent / "_cache"
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / Path(output_path).name
    results_df.to_csv(cache_path, index=False)
    logger.info(f"[metrics] Full results → {cache_path}")

    # ── Save nonllm_{tag}.csv in _cache/ (combine-friendly JSON metrics blob) ─
    if tag:
        nonllm_path = cache_dir / nonllm_file(tag)
        nonllm_rows = []
        for _, r in results_df.iterrows():
            metrics_blob = {
                "cosine_sim_same":  _safe_float(r.get(EVAL_COSINE_SAME_PROMPT)),
                "cosine_sim_multi": _safe_float(r.get(EVAL_COSINE_MULTI_REF)),
                "rouge_l_same":     _safe_float(r.get(EVAL_ROUGEL_SAME)),
                "rouge_l_multi":    _safe_float(r.get(EVAL_ROUGEL_MULTI)),
                "bertscore_f1_same":  _safe_float(r.get(EVAL_BERTSCORE_SAME)),
                "bertscore_f1_multi": _safe_float(r.get(EVAL_BERTSCORE_MULTI)),
            }
            nonllm_rows.append({
                "cluster_id":      r[PRED_CLUSTER_ID],
                "prompt_id":       r[PRED_PROMPT_ID],
                "model":           r.get("model", tag),
                "generated_label": r[PRED_GENERATED_LABEL],
                "nonllm_metrics":  json.dumps(metrics_blob),
            })
        pd.DataFrame(nonllm_rows).to_csv(nonllm_path, index=False)
        logger.info(f"[metrics] Non-LLM metrics (JSON cache) → {nonllm_path}")

    _log_summary(results_df, fine_tuned, eval_label)
    return results_df


def _safe_float(v) -> float | None:
    """Return float or None if NaN/None."""
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ── Individual metric functions ───────────────────────────────────────────────

def cosine_similarity(generated: str, reference: str, embed_model=None) -> float:
    """
    Semantic cosine similarity between generated and reference labels.
    Higher = more semantically similar. Range: [−1, 1], typically [0, 1].
    """
    if not generated or not reference:
        return 0.0
    from sentence_transformers import util
    emb_gen = embed_model.encode(generated, convert_to_tensor=True)
    emb_ref = embed_model.encode(reference, convert_to_tensor=True)
    return float(util.cos_sim(emb_gen, emb_ref).item())


def rouge_l(generated: str, reference: str, scorer=None) -> float:
    """ROUGE-L F1 score between generated and reference."""
    if not generated or not reference:
        return 0.0
    score = scorer.score(reference, generated)
    return score["rougeL"].fmeasure


def max_over_refs(
    generated: str,
    references: list[str],
    metric_fn,
    *metric_args,
) -> float:
    """
    Multi-reference evaluation: compute metric against each reference,
    return the maximum score.
    """
    if not references:
        return 0.0
    scores = [metric_fn(generated, ref, *metric_args) for ref in references if ref]
    return max(scores) if scores else 0.0


# ── Private helpers ───────────────────────────────────────────────────────────

def _load_embed_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    logger.info(f"[metrics] Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def _load_rouge():
    from rouge_score import rouge_scorer as rs
    return rs.RougeScorer(["rougeL"], use_stemmer=True)


def _build_teacher_lookup(labeled_df: pd.DataFrame, model_id: str) -> dict:
    """Build {cluster_id: {prompt_id: label}} from labeled DataFrame."""
    from phase1.data.schema import CLUSTER_ID, PROMPT_IDS, cluster_name_col
    import pandas as pd

    lookup = {}
    for cid, grp in labeled_df.groupby(CLUSTER_ID):
        row = grp.iloc[0]
        lookup[int(cid)] = {
            pid: str(row[cluster_name_col(model_id, pid)])
            for pid in PROMPT_IDS
            if cluster_name_col(model_id, pid) in row.index
            and pd.notna(row[cluster_name_col(model_id, pid)])
        }
    return lookup


def _add_bertscore(
    results_df: pd.DataFrame,
    bertscore_model: str,
    labeled_df: pd.DataFrame,
    model_id: str,
    run_bertscore: bool = False,
) -> pd.DataFrame:
    """
    Compute BERTScore and add columns to results_df.

    Loads the scoring model ONCE using BERTScorer, then batches all
    same-prompt and multi-reference computations in two passes.
    This avoids the 500+ model reload problem caused by using bert_score.score()
    per prediction.
    """
    from phase1.data.schema import EVAL_BERTSCORE_SAME, EVAL_BERTSCORE_MULTI

    if not run_bertscore:
        logger.info(
            "[metrics] BERTScore skipped (run_bertscore: false in config). "
            "Set run_bertscore: true in phase1_config.yaml to enable on Colab."
        )
        results_df[EVAL_BERTSCORE_SAME]  = None
        results_df[EVAL_BERTSCORE_MULTI] = None
        return results_df

    from bert_score import BERTScorer
    from phase1.data.schema import PRED_CLUSTER_ID, PRED_PROMPT_ID, PRED_GENERATED_LABEL

    logger.info(f"[metrics] Computing BERTScore with {bertscore_model} (loading model once) ...")
    teacher_lookup = _build_teacher_lookup(labeled_df, model_id)
    device         = _bert_score_device()

    try:
        # Load roberta-large (or configured model) once for the entire evaluation
        scorer = BERTScorer(
            model_type=bertscore_model,
            device=device,
            verbose=False,
        )
        logger.info(f"[metrics] BERTScorer loaded on {device}. Scoring {len(results_df)} predictions ...")

        same_scores  = []
        multi_scores = []

        for _, row in results_df.iterrows():
            cid = int(row[PRED_CLUSTER_ID])
            pid = row[PRED_PROMPT_ID]
            gen = str(row[PRED_GENERATED_LABEL])

            cluster_refs = teacher_lookup.get(cid, {})
            same_ref     = cluster_refs.get(pid, "")
            all_refs     = [v for v in cluster_refs.values() if v]

            # Same-prompt score: one call, no model reload
            if same_ref:
                _, _, F = scorer.score([gen], [same_ref])
                same_scores.append(float(F.mean()))
            else:
                same_scores.append(0.0)

            # Multi-reference score: one call per reference, no model reload
            if all_refs:
                ref_scores = [float(scorer.score([gen], [ref])[2].mean())
                              for ref in all_refs]
                multi_scores.append(max(ref_scores))
            else:
                multi_scores.append(0.0)

        results_df[EVAL_BERTSCORE_SAME]  = same_scores
        results_df[EVAL_BERTSCORE_MULTI] = multi_scores
        logger.info("[metrics] BERTScore complete.")

    except (OverflowError, RuntimeError, Exception) as e:
        logger.warning(
            f"[metrics] BERTScore failed: {type(e).__name__}: {e}\n"
            "Setting BERTScore columns to None."
        )
        results_df[EVAL_BERTSCORE_SAME]  = None
        results_df[EVAL_BERTSCORE_MULTI] = None

    return results_df


def _bert_score_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _log_summary(results_df: pd.DataFrame, fine_tuned: bool, eval_label: str = None) -> None:
    from phase1.data.schema import (
        EVAL_COSINE_SAME_PROMPT, EVAL_COSINE_MULTI_REF,
        EVAL_BERTSCORE_SAME, EVAL_BERTSCORE_MULTI,
        EVAL_ROUGEL_SAME, EVAL_ROUGEL_MULTI,
    )

    def _fmt(col: str) -> str:
        """Format a metric mean, showing 'skipped' if all values are None."""
        if col not in results_df.columns or results_df[col].isna().all():
            return "skipped"
        return f"{results_df[col].mean():.4f}"

    label = (
        eval_label.upper() if eval_label
        else ("FINE-TUNED (post-distillation)" if fine_tuned else "BASELINE (pre-distillation)")
    )
    logger.info(
        f"\n{'=' * 55}\n"
        f"  {label} EVALUATION SUMMARY\n"
        f"{'=' * 55}\n"
        f"  Cosine sim (same prompt):   {_fmt(EVAL_COSINE_SAME_PROMPT)}\n"
        f"  Cosine sim (multi-ref max): {_fmt(EVAL_COSINE_MULTI_REF)}\n"
        f"  BERTScore F1 (same prompt): {_fmt(EVAL_BERTSCORE_SAME)}\n"
        f"  BERTScore F1 (multi-ref):   {_fmt(EVAL_BERTSCORE_MULTI)}\n"
        f"  ROUGE-L (same prompt):      {_fmt(EVAL_ROUGEL_SAME)}\n"
        f"  ROUGE-L (multi-ref max):    {_fmt(EVAL_ROUGEL_MULTI)}\n"
        f"{'=' * 55}"
    )
