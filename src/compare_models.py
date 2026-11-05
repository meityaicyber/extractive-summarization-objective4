#!/usr/bin/env python3
"""
Comprehensive Model Comparison for Extractive Summarization of Cybersecurity Audit Reports.

Compares:
  1. Google Gemma (Gemma-4-31B)
  2. Qwen (Qwen-2.5-32B)
  3. NVIDIA Nemotron (Llama-3.1-Nemotron-70B)
  4. Heuristic Baseline (Deterministic rule-based baseline)

Evaluates on:
  - ROUGE-1 (F1, Precision, Recall)
  - ROUGE-2 (F1, Precision, Recall)
  - ROUGE-L (F1, Precision, Recall)
  - High-Severity Finding Coverage
  - Compression Ratio
  - Verbatim Fidelity (100% verbatim assertion)
  - Execution Latency (seconds/report)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure summarizer is in sys.path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from summarizer.evaluation import evaluate_summary
from summarizer.json_report import extract_report_sentences, ReportDocument
from summarizer.llm_client import GemmaClient, OllamaClient, get_llm_client
from summarizer.llm_summary import llm_extractive_paragraphs
from summarizer.selection import heuristic_paragraph_summary, verify_verbatim_summary

logger = logging.getLogger("compare_models")

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "data2")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Default model identifiers
MODEL_SPECS = {
    "gemma": "/home/aicloud/chatbot/ragengine/models/gemma-4-31B",
    "qwen": "qwen2.5:32b",
    "nemotron": "nemotron",
}


def load_evaluation_files(data_dir: str, limit: Optional[int] = None) -> List[Tuple[str, str]]:
    """Find files that contain valid reference narratives for ROUGE evaluation.

    Returns:
        List of (file_path, reference_text)
    """
    import glob

    all_files = sorted(
        glob.glob(os.path.join(data_dir, "*.json")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        if os.path.splitext(os.path.basename(p))[0].isdigit()
        else p,
    )

    eval_items = []
    for f in all_files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            narr = data.get("executive_summary", {}).get("narrative", "")
            if narr and len(narr.strip().split()) >= 30:
                eval_items.append((f, narr.strip()))
        except Exception as e:
            logger.warning("Could not read %s: %e", f, e)

    if limit:
        eval_items = eval_items[:limit]

    return eval_items


def evaluate_single_model(
    model_key: str,
    client: Any,
    eval_files: List[Tuple[str, str]],
    target_words: int = 250,
) -> Tuple[List[Dict[str, Any]], Dict[str, Tuple[float, float]]]:
    """Run evaluation for one model across all test files."""
    per_file_results: List[Dict[str, Any]] = []

    print(f"\nEvaluating Model: {model_key.upper()} ({getattr(client, 'name', 'Heuristic')})")
    print(f"{'#':<3} {'File':<12} {'Words':<8} {'R1-F1':<8} {'R1-Prec':<8} {'R1-Rec':<8} {'R2-F1':<8} {'RL-F1':<8} {'HighCov':<8} {'Time(s)':<8}")
    print("-" * 88)

    for idx, (fpath, ref_text) in enumerate(eval_files, start=1):
        fname = os.path.basename(fpath)
        with open(fpath, "r", encoding="utf-8") as fp:
            data = json.load(fp)

        doc = extract_report_sentences(data, file_path=fpath)
        t0 = time.time()

        if model_key == "baseline":
            summary_text, selected_sentences = heuristic_paragraph_summary(doc, max_words=target_words)
            elapsed = time.time() - t0
        else:
            try:
                summary_text, selected_sentences, meta = llm_extractive_paragraphs(client, doc, target_words=target_words)
                elapsed = time.time() - t0
            except Exception as e:
                logger.error("Error summarizing %s with %s: %s", fname, model_key, e)
                # Fallback to heuristic
                summary_text, selected_sentences = heuristic_paragraph_summary(doc, max_words=target_words)
                elapsed = time.time() - t0

        # Verbatim sanity check
        try:
            verify_verbatim_summary(selected_sentences, doc.source_text)
            verbatim_pass = True
        except AssertionError as e:
            verbatim_pass = False
            logger.warning("Verbatim check failed for %s on %s: %s", model_key, fname, e)

        # Compute metrics
        unit_map = {s.uid: s for s in doc.all_sentences}
        # Create lightweight Unit wrappers for evaluate_summary
        from summarizer.segmentation import Unit
        units = []
        uid_to_idx = {}
        for i, s in enumerate(doc.all_sentences):
            u = Unit(
                uid=s.uid,
                text=s.text,
                page=1,
                section_index=0,
                section_title=s.section,
                section_type=s.role,
                finding_id=s.finding_id,
                is_table_row=False,
                words=s.words,
                eff_severity=s.severity_score,
            )
            units.append(u)
            uid_to_idx[s.uid] = i

        selected_indices = [uid_to_idx[s.uid] for s in selected_sentences if s.uid in uid_to_idx]

        metrics = evaluate_summary(
            units=units,
            selected_ids=selected_indices,
            reference_text=ref_text,
            total_words=doc.total_words,
        )

        word_count = sum(s.words for s in selected_sentences)

        record = {
            "model": model_key,
            "file": fname,
            "words": word_count,
            "time_s": round(elapsed, 2),
            "verbatim_verified": verbatim_pass,
            "rouge1_f1": metrics.get("rouge1") or 0.0,
            "rouge1_precision": metrics.get("rouge1_precision") or 0.0,
            "rouge1_recall": metrics.get("rouge1_recall") or 0.0,
            "rouge2_f1": metrics.get("rouge2") or 0.0,
            "rouge2_precision": metrics.get("rouge2_precision") or 0.0,
            "rouge2_recall": metrics.get("rouge2_recall") or 0.0,
            "rougeL_f1": metrics.get("rougeL") or 0.0,
            "rougeL_precision": metrics.get("rougeL_precision") or 0.0,
            "rougeL_recall": metrics.get("rougeL_recall") or 0.0,
            "high_finding_coverage": metrics.get("high_finding_coverage") or 0.0,
            "compression": metrics.get("compression") or 0.0,
            "summary_text": summary_text,
        }
        per_file_results.append(record)

        print(
            f"{idx:<3} {fname:<12} {word_count:<8} "
            f"{record['rouge1_f1']:<8.4f} {record['rouge1_precision']:<8.4f} {record['rouge1_recall']:<8.4f} "
            f"{record['rouge2_f1']:<8.4f} {record['rougeL_f1']:<8.4f} "
            f"{record['high_finding_coverage']:<8.2f} {elapsed:<8.2f}"
        )

    # Compute aggregate stats (mean ± std)
    metric_keys = [
        "rouge1_f1", "rouge1_precision", "rouge1_recall",
        "rouge2_f1", "rouge2_precision", "rouge2_recall",
        "rougeL_f1", "rougeL_precision", "rougeL_recall",
        "high_finding_coverage", "compression", "time_s",
    ]
    aggregate: Dict[str, Tuple[float, float]] = {}
    for k in metric_keys:
        vals = [r[k] for r in per_file_results]
        aggregate[k] = (float(np.mean(vals)), float(np.std(vals)))

    return per_file_results, aggregate


def format_markdown_table(all_aggregates: Dict[str, Dict[str, Tuple[float, float]]]) -> str:
    """Format aggregate results into a clear GitHub Markdown table."""
    headers = [
        "Model",
        "ROUGE-1 (F1)",
        "ROUGE-1 (P / R)",
        "ROUGE-2 (F1)",
        "ROUGE-2 (P / R)",
        "ROUGE-L (F1)",
        "ROUGE-L (P / R)",
        "High-Sev Cov",
        "Compress",
        "Latency (s)",
    ]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]

    for model_name, agg in all_aggregates.items():
        r1_f1 = f"{agg['rouge1_f1'][0]:.4f} ± {agg['rouge1_f1'][1]:.3f}"
        r1_pr = f"{agg['rouge1_precision'][0]:.3f} / {agg['rouge1_recall'][0]:.3f}"
        r2_f1 = f"{agg['rouge2_f1'][0]:.4f} ± {agg['rouge2_f1'][1]:.3f}"
        r2_pr = f"{agg['rouge2_precision'][0]:.3f} / {agg['rouge2_recall'][0]:.3f}"
        rl_f1 = f"{agg['rougeL_f1'][0]:.4f} ± {agg['rougeL_f1'][1]:.3f}"
        rl_pr = f"{agg['rougeL_precision'][0]:.3f} / {agg['rougeL_recall'][0]:.3f}"
        cov = f"{agg['high_finding_coverage'][0]*100:.1f}%"
        comp = f"{agg['compression'][0]*100:.1f}%"
        lat = f"{agg['time_s'][0]:.2f}s"

        row = [
            f"**{model_name}**",
            r1_f1,
            r1_pr,
            r2_f1,
            r2_pr,
            rl_f1,
            rl_pr,
            cov,
            comp,
            lat,
        ]
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare Gemma, Qwen, and Nemotron on Extractive Summarization")
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR, help="Path to JSON reports")
    parser.add_argument("--limit", type=int, default=10, help="Number of evaluation documents to use")
    parser.add_argument("--out_dir", default=RESULTS_DIR, help="Results output directory")
    parser.add_argument("--gemma_path", default=MODEL_SPECS["gemma"], help="Local Gemma model folder")
    parser.add_argument("--qwen_name", default=MODEL_SPECS["qwen"], help="Qwen Ollama model name")
    parser.add_argument("--nemotron_name", default=MODEL_SPECS["nemotron"], help="Nemotron Ollama model name")
    parser.add_argument("--models", nargs="+", default=["gemma", "qwen", "nemotron", "baseline"], help="Models to benchmark")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 80)
    print("EXTRACTIVE SUMMARIZATION MODEL BENCHMARK & COMPARISON")
    print(f"Data directory: {args.data_dir}")
    print(f"Evaluation limit: {args.limit} reports with gold reference narratives")
    print(f"Models scheduled: {args.models}")
    print("=" * 80)

    eval_files = load_evaluation_files(args.data_dir, limit=args.limit)
    print(f"Loaded {len(eval_files)} evaluation reports with reference narratives:\n" +
          ", ".join(os.path.basename(f[0]) for f in eval_files))

    all_per_file: Dict[str, List[Dict[str, Any]]] = {}
    all_aggregates: Dict[str, Dict[str, Tuple[float, float]]] = {}

    for m in args.models:
        client = None
        if m == "gemma":
            print(f"\nInitializing Gemma model from {args.gemma_path} ...", flush=True)
            t_load = time.time()
            try:
                client = GemmaClient(model_path=args.gemma_path)
                print(f"Gemma loaded successfully in {time.time() - t_load:.1f}s.")
            except Exception as e:
                print(f"Error loading Gemma: {e}")
                continue
        elif m == "qwen":
            print(f"\nInitializing Qwen client ({args.qwen_name}) ...", flush=True)
            try:
                client = OllamaClient(model_name=args.qwen_name)
                client.ping()
                print(f"Connected to Qwen ({args.qwen_name}) successfully.")
            except Exception as e:
                print(f"Error connecting to Qwen: {e}")
                continue
        elif m == "nemotron":
            print(f"\nInitializing Nemotron client ({args.nemotron_name}) ...", flush=True)
            try:
                client = OllamaClient(model_name=args.nemotron_name)
                client.ping()
                print(f"Connected to Nemotron ({args.nemotron_name}) successfully.")
            except Exception as e:
                print(f"Error connecting to Nemotron: {e}")
                continue
        elif m == "baseline":
            client = None
        else:
            print(f"Unknown model: {m}")
            continue

        per_file, agg = evaluate_single_model(m, client, eval_files)
        all_per_file[m] = per_file
        all_aggregates[m] = agg

        # Free GPU memory if Gemma
        if m == "gemma" and client is not None:
            del client
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print summary table
    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 80)
    md_table = format_markdown_table(all_aggregates)
    print(md_table)

    # Save detailed outputs
    out_json = os.path.join(args.out_dir, "model_comparison_results.json")
    with open(out_json, "w", encoding="utf-8") as fp:
        json.dump({
            "aggregates": {k: {mk: {"mean": mv[0], "std": mv[1]} for mk, mv in v.items()} for k, v in all_aggregates.items()},
            "details": all_per_file,
        }, fp, indent=2)
    print(f"\nDetailed evaluation results saved to: {out_json}")

    out_md = os.path.join(args.out_dir, "model_comparison_table.md")
    with open(out_md, "w", encoding="utf-8") as fp:
        fp.write("# Extractive Summarization Model Comparison\n\n" + md_table + "\n")
    print(f"Markdown table saved to: {out_md}")


if __name__ == "__main__":
    main()
