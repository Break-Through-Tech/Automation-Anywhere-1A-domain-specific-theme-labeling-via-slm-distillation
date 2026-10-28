"""
analysis/stats_tests.py

Statistical significance testing for SLM distillation results.

Answers the question the summary CSVs cannot: is a reported gain real,
or is it noise? Reads the PER-PREDICTION pivot files written by
phase1/evaluation/combine.py and runs the appropriate test for each
comparison.

Usage
-----
    python analysis/stats_tests.py --run_dir outputs/20260929_2353_Llama-3.2-3B-Instruct_ep5
    python analysis/stats_tests.py --run_dir <dir> --split test --out results/stats/

Required inputs (written automatically by combine.py into <run_dir>/evaluation/):
    nonllm_by_cluster.csv    cosine / ROUGE-L / BERTScore per (cluster, prompt, model)
    llm_by_cluster.csv       judge scores per (cluster, prompt, model)
    latency_by_cluster.csv   per-request latency per model   (optional)

NOTE: metrics_summary.csv and judge_summary.csv are MEANS ONLY and cannot be
used here. A significance test needs the individual observations.

Which test, and why
-------------------
baseline vs fine-tuned (quality)
    Paired data: both models are scored on the SAME (cluster, prompt) items.
    Pairing removes between-item variance, so it is far more sensitive than
    an unpaired test. Scores are bounded [0,1] or [1,5], often skewed, and n
    is small, so normality is not assumed.
    -> Wilcoxon signed-rank (primary), paired t-test (secondary),
       rank-biserial correlation (effect size), win rate (plain-English).

teacher vs baseline vs fine-tuned (latency)
    Three independent groups with very unequal variances (observed SD ratio
    >5x in the team's benchmark). Classic one-way ANOVA assumes equal
    variance and is NOT valid here.
    -> Welch's ANOVA (primary), Kruskal-Wallis (non-parametric check),
       pairwise Welch t-tests with Holm correction, Cohen's d.

Multiple comparisons
--------------------
Testing many metrics inflates the false-positive rate. Holm-Bonferroni is
applied across the metric family and reported alongside the raw p-value.
"""

from __future__ import annotations   # allows `X | None` hints on Python 3.9

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ALPHA = 0.05
TAGS = ["teacher", "baseline", "finetuned"]


# ── Multiple-comparison correction ────────────────────────────────────────────

def holm_bonferroni(pvals: list[float]) -> list[float]:
    """
    Holm-Bonferroni step-down adjusted p-values.

    Uniformly more powerful than plain Bonferroni while controlling the same
    family-wise error rate. Returns adjusted p-values in the input order.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)          # enforce monotonicity
        adj[idx] = min(running, 1.0)
    return adj.tolist()


# ── Effect sizes ──────────────────────────────────────────────────────────────

def rank_biserial(diffs: np.ndarray) -> float:
    """
    Effect size for the Wilcoxon signed-rank test, in [-1, 1].

    Computed as the normalised difference between the sum of positive ranks
    and the sum of negative ranks. Zero differences are dropped, matching
    the Wilcoxon test itself. ~0.1 small, ~0.3 medium, ~0.5 large.
    """
    d = diffs[diffs != 0]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d))
    r_plus = ranks[d > 0].sum()
    r_minus = ranks[d < 0].sum()
    total = ranks.sum()
    return float((r_plus - r_minus) / total)


def cohens_d_pooled(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples, pooled SD. |d|>0.8 = large."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    sp = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if sp == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / sp)


# ── Welch's ANOVA ─────────────────────────────────────────────────────────────

def welch_anova(groups: list[np.ndarray]) -> tuple[float, float, float]:
    """
    Welch's one-way ANOVA (does NOT assume equal variances).

    Returns (F, df2, p). Use this instead of scipy.stats.f_oneway whenever
    group variances differ materially, which is the norm for latency data.
    """
    k = len(groups)
    n = np.array([len(g) for g in groups], dtype=float)
    m = np.array([g.mean() for g in groups], dtype=float)
    v = np.array([g.var(ddof=1) for g in groups], dtype=float)

    w = n / v                                    # inverse-variance weights
    sw = w.sum()
    mbar = (w * m).sum() / sw

    num = ((w * (m - mbar) ** 2).sum()) / (k - 1)
    lam = (((1 - w / sw) ** 2) / (n - 1)).sum()
    denom = 1 + (2 * (k - 2) / (k ** 2 - 1)) * lam
    F = num / denom

    df1 = k - 1
    df2 = (k ** 2 - 1) / (3 * lam)
    p = stats.f.sf(F, df1, df2)
    return float(F), float(df2), float(p)


def welch_ttest(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Welch's unequal-variance t-test. Returns (t, df, p)."""
    t, p = stats.ttest_ind(a, b, equal_var=False)
    n1, n2 = len(a), len(b)
    s1, s2 = a.var(ddof=1), b.var(ddof=1)
    df = (s1 / n1 + s2 / n2) ** 2 / (
        (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
    )
    return float(t), float(df), float(p)


# ── Paired comparison: baseline vs fine-tuned ─────────────────────────────────

def paired_report(df: pd.DataFrame, metric: str,
                  a_tag: str = "baseline", b_tag: str = "finetuned") -> dict | None:
    """
    Compare two models on the same items for one metric.

    Rows where either model is missing are dropped pairwise, so one model's
    gaps never silently shrink the other's sample.
    """
    ca, cb = f"{metric}_{a_tag}", f"{metric}_{b_tag}"
    if ca not in df.columns or cb not in df.columns:
        return None

    pair = df[[ca, cb]].dropna()
    n = len(pair)
    if n < 3:
        return {"metric": metric, "n": n, "skipped": "n < 3, no test possible"}

    a = pair[ca].to_numpy(dtype=float)
    b = pair[cb].to_numpy(dtype=float)
    diffs = b - a                                 # positive = fine-tuned better

    # Wilcoxon is undefined when every difference is zero.
    if np.all(diffs == 0):
        return {"metric": metric, "n": n, "skipped": "all differences are zero"}

    try:
        w_stat, w_p = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError as e:
        w_stat, w_p = float("nan"), float("nan")
        print(f"  [warn] Wilcoxon failed for {metric}: {e}", file=sys.stderr)

    t_stat, t_p = stats.ttest_rel(b, a)

    nonzero = diffs[diffs != 0]
    wins = int((diffs > 0).sum())

    return {
        "metric": metric,
        "n": n,
        f"mean_{a_tag}": float(a.mean()),
        f"mean_{b_tag}": float(b.mean()),
        "mean_gain": float(diffs.mean()),
        "median_gain": float(np.median(diffs)),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p": float(w_p),
        "ttest_p": float(t_p),
        "rank_biserial": rank_biserial(diffs),
        "win_rate": wins / len(nonzero) if len(nonzero) else float("nan"),
        "wins": wins,
        "losses": int((diffs < 0).sum()),
        "ties": int((diffs == 0).sum()),
    }


# ── Latency comparison across three models ────────────────────────────────────

def latency_report(lat: pd.DataFrame) -> dict | None:
    """Three-group latency comparison with unequal-variance handling."""
    cols = {t: f"{t}_latency_s" for t in TAGS if f"{t}_latency_s" in lat.columns}
    groups = {t: lat[c].dropna().to_numpy(dtype=float) for t, c in cols.items()}
    groups = {t: g for t, g in groups.items() if len(g) >= 2}

    if len(groups) < 2:
        return None

    sds = {t: float(g.std(ddof=1)) for t, g in groups.items()}
    nz = [s for s in sds.values() if s > 0]
    ratio = (max(nz) / min(nz)) if len(nz) >= 2 else float("nan")

    out = {
        "n_per_group": {t: int(len(g)) for t, g in groups.items()},
        "mean_s": {t: float(g.mean()) for t, g in groups.items()},
        "median_s": {t: float(np.median(g)) for t, g in groups.items()},
        "sd_s": sds,
        "p95_s": {t: float(np.percentile(g, 95)) for t, g in groups.items()},
        "sd_ratio": ratio,
        "equal_variance_ok": bool(ratio < 2.0) if np.isfinite(ratio) else None,
    }

    glist = list(groups.values())

    if len(groups) >= 3:
        F, df2, p = welch_anova(glist)
        out["welch_anova"] = {"F": F, "df2": df2, "p": p}
        h, hp = stats.kruskal(*glist)
        out["kruskal_wallis"] = {"H": float(h), "p": float(hp)}

    # Levene's test formally checks the equal-variance assumption.
    if len(groups) >= 2:
        lev_stat, lev_p = stats.levene(*glist, center="median")
        out["levene"] = {"stat": float(lev_stat), "p": float(lev_p)}

    names = list(groups)
    pairs, raw_p = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            t, df, p = welch_ttest(groups[x], groups[y])
            pairs.append({
                "a": x, "b": y,
                "mean_diff_s": float(groups[x].mean() - groups[y].mean()),
                "t": t, "df": df, "p_raw": p,
                "cohens_d": cohens_d_pooled(groups[x], groups[y]),
            })
            raw_p.append(p)

    for pr, adj in zip(pairs, holm_bonferroni(raw_p)):
        pr["p_holm"] = adj
        pr["significant"] = bool(adj < ALPHA)
    out["pairwise_welch"] = pairs
    return out


# ── Threshold classification view ─────────────────────────────────────────────

def classification_report(df: pd.DataFrame, threshold: float = 4.0,
                          dim: str = "composite") -> dict | None:
    """
    Recast judge scores as a binary "acceptable label" classification.

    Treats the teacher as the reference standard: teacher >= threshold is the
    positive class, and the student's own score is its prediction. This gives
    precision / recall / F1 / confusion matrix, which read far more naturally
    to a stakeholder than "cosine 0.84".

    Requires teacher judge scores, so it only works under reference-free
    judging. In reference mode the teacher is hardcoded to 5.0 and every
    metric here degenerates.
    """
    tcol = f"{dim}_teacher"
    if tcol not in df.columns or df[tcol].dropna().empty:
        return None
    if float(df[tcol].dropna().nunique()) == 1 and float(df[tcol].dropna().iloc[0]) == 5.0:
        return {"skipped": "teacher scores are hardcoded 5.0 (reference mode) "
                           "— rerun with reference-free judging"}

    out = {"threshold": threshold, "dimension": dim, "models": {}}
    for tag in ("baseline", "finetuned"):
        scol = f"{dim}_{tag}"
        if scol not in df.columns:
            continue
        pair = df[[tcol, scol]].dropna()
        if pair.empty:
            continue
        y_true = (pair[tcol].to_numpy(dtype=float) >= threshold)
        y_pred = (pair[scol].to_numpy(dtype=float) >= threshold)

        tp = int((y_true & y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())

        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if np.isfinite(prec) and np.isfinite(rec) and (prec + rec) else float("nan"))

        out["models"][tag] = {
            "n": int(len(pair)),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
            "precision": prec, "recall": rec, "f1": f1,
            "accuracy": (tp + tn) / len(pair),
            "pass_rate": float(y_pred.mean()),
        }

    # McNemar: did the two models differ on the SAME items? (paired, binary)
    bcol, fcol = f"{dim}_baseline", f"{dim}_finetuned"
    if bcol in df.columns and fcol in df.columns:
        pair = df[[tcol, bcol, fcol]].dropna()
        if len(pair) >= 3:
            yt = pair[tcol].to_numpy(dtype=float) >= threshold
            pb = pair[bcol].to_numpy(dtype=float) >= threshold
            pf = pair[fcol].to_numpy(dtype=float) >= threshold
            b_only = int(((pb == yt) & (pf != yt)).sum())   # baseline right, ft wrong
            f_only = int(((pf == yt) & (pb != yt)).sum())   # ft right, baseline wrong
            if (b_only + f_only) > 0:
                # Exact binomial McNemar — correct for the small counts seen here.
                p = float(stats.binomtest(f_only, b_only + f_only, 0.5).pvalue)
            else:
                p = float("nan")
            out["mcnemar"] = {
                "finetuned_only_correct": f_only,
                "baseline_only_correct": b_only,
                "discordant": b_only + f_only,
                "p_exact": p,
            }
    return out


# ── Orchestration ─────────────────────────────────────────────────────────────

NONLLM_METRICS = ["cosine_sim_same", "cosine_sim_multi",
                  "rouge_l_same", "rouge_l_multi",
                  "bertscore_f1_same", "bertscore_f1_multi"]
JUDGE_DIMS = ["equivalence", "faithfulness", "specificity", "composite"]


def _load(run_dir: Path, name: str) -> pd.DataFrame | None:
    for cand in (run_dir / "evaluation" / name, run_dir / name):
        if cand.exists():
            return pd.read_csv(cand)
    return None


def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    if split == "all" or "split" not in df.columns:
        return df
    return df[df["split"] == split]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True,
                    help="Run directory containing evaluation/ pivot CSVs")
    ap.add_argument("--split", default="test",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--threshold", type=float, default=4.0,
                    help="Judge score at or above which a label counts as acceptable")
    ap.add_argument("--out", default=None, help="Directory to write results")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 1

    nonllm = _load(run_dir, "nonllm_by_cluster.csv")
    judge = _load(run_dir, "llm_by_cluster.csv")
    latency = _load(run_dir, "latency_by_cluster.csv")

    if nonllm is None and judge is None and latency is None:
        print(
            "ERROR: no per-prediction pivot files found.\n"
            "  Looked for nonllm_by_cluster.csv / llm_by_cluster.csv / "
            "latency_by_cluster.csv\n"
            f"  under {run_dir}/evaluation/ and {run_dir}/\n\n"
            "  metrics_summary.csv and judge_summary.csv contain MEANS ONLY and\n"
            "  cannot support a significance test. Copy the by_cluster files out\n"
            "  of your Colab Drive run folder.",
            file=sys.stderr,
        )
        return 2

    report: dict = {"run_dir": str(run_dir), "split": args.split, "alpha": ALPHA}
    print(f"\n{'=' * 72}\n  STATISTICAL TESTS — {run_dir.name}  (split: {args.split})\n{'=' * 72}")

    # ── Paired quality tests ──────────────────────────────────────────────────
    for src, metrics, title in ((nonllm, NONLLM_METRICS, "SIMILARITY METRICS"),
                                (judge, JUDGE_DIMS, "LLM JUDGE SCORES")):
        if src is None:
            continue
        df = _filter_split(src, args.split)
        if df.empty:
            print(f"\n[{title}] no rows for split={args.split} — skipped")
            continue

        rows = [r for m in metrics if (r := paired_report(df, m)) is not None]
        tested = [r for r in rows if "skipped" not in r]
        for r, adj in zip(tested, holm_bonferroni([r["wilcoxon_p"] for r in tested])):
            r["wilcoxon_p_holm"] = adj
            r["significant"] = bool(adj < ALPHA)

        print(f"\n{'─' * 72}\n  {title} — baseline vs fine-tuned (paired)\n{'─' * 72}")
        print(f"{'metric':20s} {'n':>3s} {'base':>7s} {'ft':>7s} {'gain':>8s} "
              f"{'win%':>6s} {'p':>9s} {'p_holm':>9s} {'effect':>7s}")
        for r in rows:
            if "skipped" in r:
                print(f"{r['metric']:20s} {r['n']:3d}  -- {r['skipped']}")
                continue
            star = " *" if r["significant"] else ""
            print(f"{r['metric']:20s} {r['n']:3d} {r['mean_baseline']:7.4f} "
                  f"{r['mean_finetuned']:7.4f} {r['mean_gain']:+8.4f} "
                  f"{r['win_rate'] * 100:5.1f}% {r['wilcoxon_p']:9.4f} "
                  f"{r['wilcoxon_p_holm']:9.4f} {r['rank_biserial']:+7.3f}{star}")
        report[title.lower().replace(" ", "_")] = rows

    # ── Latency ───────────────────────────────────────────────────────────────
    if latency is not None:
        lat = _filter_split(latency, args.split)
        lr = latency_report(lat) if not lat.empty else None
        if lr:
            print(f"\n{'─' * 72}\n  LATENCY — three-model comparison\n{'─' * 72}")
            for t in TAGS:
                if t in lr["mean_s"]:
                    print(f"  {t:12s} n={lr['n_per_group'][t]:3d}  "
                          f"mean={lr['mean_s'][t]:.4f}s  sd={lr['sd_s'][t]:.4f}  "
                          f"p95={lr['p95_s'][t]:.4f}s")
            print(f"\n  SD ratio = {lr['sd_ratio']:.2f}x  "
                  f"(>2x means classic ANOVA is invalid)")
            if "levene" in lr:
                print(f"  Levene equal-variance test: p = {lr['levene']['p']:.4g}")
            if "welch_anova" in lr:
                w = lr["welch_anova"]
                print(f"  Welch ANOVA: F = {w['F']:.3f}, df2 = {w['df2']:.1f}, "
                      f"p = {w['p']:.4g}")
                print(f"  Kruskal-Wallis: H = {lr['kruskal_wallis']['H']:.3f}, "
                      f"p = {lr['kruskal_wallis']['p']:.4g}")
            print("\n  Pairwise (Welch t, Holm-corrected):")
            for pr in lr["pairwise_welch"]:
                print(f"    {pr['a']:11s} vs {pr['b']:11s} "
                      f"diff={pr['mean_diff_s']:+.4f}s  d={pr['cohens_d']:+.2f}  "
                      f"p_holm={pr['p_holm']:.4g}  "
                      f"{'SIG' if pr['significant'] else 'n.s.'}")
            report["latency"] = lr

    # ── Classification view ───────────────────────────────────────────────────
    if judge is not None:
        cr = classification_report(_filter_split(judge, args.split), args.threshold)
        if cr:
            print(f"\n{'─' * 72}\n  CLASSIFICATION VIEW "
                  f"(score >= {args.threshold} = acceptable)\n{'─' * 72}")
            if "skipped" in cr:
                print(f"  skipped: {cr['skipped']}")
            else:
                for tag, m in cr["models"].items():
                    c = m["confusion"]
                    print(f"  {tag:10s} P={m['precision']:.3f} R={m['recall']:.3f} "
                          f"F1={m['f1']:.3f} acc={m['accuracy']:.3f}  "
                          f"[tp={c['tp']} fp={c['fp']} fn={c['fn']} tn={c['tn']}]")
                if "mcnemar" in cr:
                    mc = cr["mcnemar"]
                    print(f"\n  McNemar (paired, exact): ft-only-correct="
                          f"{mc['finetuned_only_correct']}, baseline-only-correct="
                          f"{mc['baseline_only_correct']}, p={mc['p_exact']:.4g}")
            report["classification"] = cr

    print(f"\n{'=' * 72}")
    print("  * = significant after Holm correction at alpha = 0.05")
    print("  A non-significant result with small n means UNDERPOWERED,")
    print("  not 'no effect'. Check the effect size and win rate too.")
    print(f"{'=' * 72}\n")

    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"stats_{run_dir.name}_{args.split}.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2, default=float)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
