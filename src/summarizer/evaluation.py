"""
Evaluation metrics, table rendering, and results export (ROUGE, coverage, fragment rate, compression, AUC).
"""

from __future__ import annotations

import csv
import json
import logging
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

try:
    from rouge_score import rouge_scorer
except ImportError:  # pragma: no cover
    rouge_scorer = None

from summarizer.labels import silver_labels
from summarizer.segmentation import Unit

logger = logging.getLogger(__name__)


def evaluate_summary(
    units: Sequence[Unit],
    selected_ids: Sequence[int],
    reference_text: Optional[str],
    total_words: int,
    high_severity_threshold: float = 0.8,
) -> Dict[str, Optional[float]]:
    """Compute all evaluation metrics for a selected summary.

    Metrics:
      - compression: total selected words / total source words
      - high_finding_coverage: fraction of high-severity findings covered
      - fragment_rate: fraction of selected units < 6 words
      - rouge1, rouge2, rougeL: F1 measures against reference summary

    Args:
        units: Sequence of Unit objects in document.
        selected_ids: Indices of selected units.
        reference_text: Reference summary text (optional).
        total_words: Total words in source document.
        high_severity_threshold: Effective severity threshold for high-severity findings.

    Returns:
        Dict of metric names to float values (or None if unavailable).
    """
    selected_units = [units[i] for i in selected_ids]
    summary_words = sum(u.words for u in selected_units)
    compression = summary_words / max(total_words, 1)

    # High severity findings coverage
    gold_high_fids = {
        u.finding_id for u in units
        if u.finding_id and u.eff_severity >= high_severity_threshold
    }
    covered_fids = {u.finding_id for u in selected_units if u.finding_id and u.words >= 8}

    if gold_high_fids:
        high_finding_coverage = len(gold_high_fids & covered_fids) / len(gold_high_fids)
    else:
        high_finding_coverage = 1.0  # vacuously covered if none exist

    # Fragment rate: fraction of selected units shorter than 6 words
    if selected_units:
        fragment_rate = sum(1 for u in selected_units if u.words < 6) / len(selected_units)
    else:
        fragment_rate = 0.0

    metrics: Dict[str, Optional[float]] = {
        "compression": round(compression, 4),
        "high_finding_coverage": round(high_finding_coverage, 4),
        "fragment_rate": round(fragment_rate, 4),
        "rouge1": None,
        "rouge1_precision": None,
        "rouge1_recall": None,
        "rouge2": None,
        "rouge2_precision": None,
        "rouge2_recall": None,
        "rougeL": None,
        "rougeL_precision": None,
        "rougeL_recall": None,
    }

    # ROUGE scoring
    if reference_text and rouge_scorer is not None:
        summary_text = "\n".join(u.text for u in selected_units)
        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(reference_text, summary_text)
        metrics["rouge1"] = round(float(scores["rouge1"].fmeasure), 4)
        metrics["rouge1_precision"] = round(float(scores["rouge1"].precision), 4)
        metrics["rouge1_recall"] = round(float(scores["rouge1"].recall), 4)
        metrics["rouge2"] = round(float(scores["rouge2"].fmeasure), 4)
        metrics["rouge2_precision"] = round(float(scores["rouge2"].precision), 4)
        metrics["rouge2_recall"] = round(float(scores["rouge2"].recall), 4)
        metrics["rougeL"] = round(float(scores["rougeL"].fmeasure), 4)
        metrics["rougeL_precision"] = round(float(scores["rougeL"].precision), 4)
        metrics["rougeL_recall"] = round(float(scores["rougeL"].recall), 4)

    return metrics


def silver_auc(
    units: Sequence[Unit],
    scores: np.ndarray,
    reference_text: str,
    threshold: float = 0.4,
) -> Optional[float]:
    """Compute ROC AUC of predicted scores against silver labels.

    Args:
        units: Sequence of Unit objects.
        scores: Predicted relevance probabilities.
        reference_text: Reference summary text.
        threshold: Silver-label unigram F1 threshold.

    Returns:
        ROC AUC float, or None if labels contain only one class.
    """
    y_true = silver_labels(units, reference_text, threshold=threshold)
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return round(float(roc_auc_score(y_true, scores)), 4)
    except Exception as e:
        logger.warning("ROC AUC computation failed: %s", e)
        return None


def format_comparison_table(
    target_name: str,
    system_metrics: Dict[str, Dict[str, Any]],
    silver_threshold: float = 0.4,
) -> str:
    """Format a clean markdown/ASCII comparison table for a target report.

    Args:
        target_name: Identifier for the target document.
        system_metrics: Mapping of system_name -> metric_dict.
        silver_threshold: Silver label threshold used for evaluation.

    Returns:
        Formatted table string.
    """
    headers = [
        "System",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "High-Sev Cov",
        "Frag Rate",
        "Compress",
        "Silver AUC",
    ]
    key_map = [
        "rouge1",
        "rouge2",
        "rougeL",
        "high_finding_coverage",
        "fragment_rate",
        "compression",
        "silver_auc",
    ]

    col_widths = [14, 10, 10, 10, 14, 11, 10, 12]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "-" * sum(col_widths)

    lines = [
        f"\nEvaluation Comparison for Target: {target_name} (Silver threshold: {silver_threshold})",
        sep_line,
        header_line,
        sep_line,
    ]

    for sys_name, m in system_metrics.items():
        row = [sys_name.ljust(col_widths[0])]
        for k, w in zip(key_map, col_widths[1:]):
            val = m.get(k)
            if val is None:
                str_val = "N/A"
            else:
                str_val = f"{val:.3f}" if isinstance(val, (int, float)) else str(val)
            row.append(str_val.ljust(w))
        lines.append("".join(row))

    lines.append(sep_line)
    return "\n".join(lines)


def format_aggregate_table(
    aggregate_stats: Dict[str, Dict[str, Tuple[float, float]]],
    silver_threshold: float = 0.4,
) -> str:
    """Format an aggregate comparison table with mean ± std for leave-one-out evaluation.

    Args:
        aggregate_stats: Mapping: system -> metric -> (mean, std).
        silver_threshold: Silver label threshold.

    Returns:
        Formatted aggregate markdown/ASCII table string.
    """
    headers = [
        "System",
        "ROUGE-1",
        "ROUGE-2",
        "ROUGE-L",
        "High-Sev Cov",
        "Frag Rate",
        "Compress",
        "Silver AUC",
    ]
    key_map = [
        "rouge1",
        "rouge2",
        "rougeL",
        "high_finding_coverage",
        "fragment_rate",
        "compression",
        "silver_auc",
    ]

    col_widths = [14, 15, 15, 15, 16, 15, 15, 15]
    header_line = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    sep_line = "=" * sum(col_widths)

    lines = [
        f"\n=== LEAVE-ONE-OUT AGGREGATE RESULTS (Mean ± Std, Silver Threshold: {silver_threshold}) ===",
        sep_line,
        header_line,
        "-" * sum(col_widths),
    ]

    for sys_name, m in aggregate_stats.items():
        row = [sys_name.ljust(col_widths[0])]
        for k, w in zip(key_map, col_widths[1:]):
            if k in m and m[k] is not None:
                mean_v, std_v = m[k]
                str_val = f"{mean_v:.3f}±{std_v:.3f}"
            else:
                str_val = "N/A"
            row.append(str_val.ljust(w))
        lines.append("".join(row))

    lines.append(sep_line)
    return "\n".join(lines)


def save_results_csv_and_json(
    detailed_results: List[Dict[str, Any]],
    aggregate_results: Dict[str, Any],
    output_dir: str = "results",
) -> Tuple[str, str]:
    """Save detailed and aggregate evaluation results to CSV and JSON in output_dir.

    Args:
        detailed_results: List of per-document, per-system evaluation rows.
        aggregate_results: Dictionary of aggregated metrics.
        output_dir: Target output directory.

    Returns:
        Tuple of (csv_path, json_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "evaluation_results.csv")
    json_path = os.path.join(output_dir, "evaluation_results.json")

    # Save CSV
    if detailed_results:
        fieldnames = list(detailed_results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in detailed_results:
                writer.writerow(row)
        logger.info("Saved CSV evaluation results to %s", csv_path)

    # Save JSON
    payload = {
        "detailed": detailed_results,
        "aggregate": aggregate_results,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("Saved JSON evaluation results to %s", json_path)

    return csv_path, json_path
