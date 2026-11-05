#!/usr/bin/env python3
"""
Batch extractive paragraph summarization of audit-report JSON files.

Produces 2-to-3 structured text paragraphs (Overview, Key Findings & Impacts,
Recommendations) where 100% of the sentences are verbatim extracts from the
source report.

Output:
  Clean text paragraph summaries (.txt), saved in:
    <out_dir>/<stem>.txt
    <out_dir>/<stem>/summary.txt

Usage:
  # Summarize a single file (e.g. data2/1.json)
  python run_json_batch.py --file ../data2/1.json

  # Summarize first 3 files
  python run_json_batch.py --limit 3

  # Full batch across all files in data2/
  python run_json_batch.py

  # Fast heuristic baseline (without loading Gemma)
  python run_json_batch.py --method baseline
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

# Add src to python path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from summarizer.json_report import extract_report_sentences, ReportDocument
from summarizer.llm_client import GemmaClient, GemmaError, OllamaClient, get_llm_client
from summarizer.llm_summary import llm_extractive_paragraphs
from summarizer.selection import heuristic_paragraph_summary, verify_verbatim_summary

PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))
DEFAULT_IN_DIR = os.path.join(PROJECT_ROOT, "data2")
DEFAULT_OUT_DIR = os.path.join(PROJECT_ROOT, "summaries")
DEFAULT_MODEL_PATH = (
    os.environ.get("GEMMA_MODEL_PATH")
    or "/home/aicloud/chatbot/ragengine/models/gemma-4-31B"
)

logger = logging.getLogger("run_json_batch")


def process_file(
    path: str,
    out_dir: str,
    client: Optional[GemmaClient],
    method: str = "gemma",
    target_words: int = 250,
    force: bool = False,
) -> Dict[str, Any]:
    """Summarize one JSON audit report into clean extractive text paragraphs.

    Args:
        path: Path to the input JSON file.
        out_dir: Output summaries folder.
        client: Loaded GemmaClient instance (if method == 'gemma').
        method: 'gemma' or 'baseline'.
        target_words: Soft target word count for the summary.
        force: Overwrite existing summary if True.

    Returns:
        Summary execution record dict.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    out_file_root = os.path.join(out_dir, f"{stem}.txt")
    out_file_sub = os.path.join(out_dir, stem, "summary.txt")

    row: Dict[str, Any] = {
        "file": os.path.basename(path),
        "stem": stem,
        "status": "",
        "paragraphs": 0,
        "words": 0,
        "time_s": 0.0,
        "method": method,
        "error": "",
    }

    if os.path.isfile(out_file_root) and not force:
        try:
            with open(out_file_root, encoding="utf-8") as f:
                content = f.read().strip()
            paras = [p for p in content.split("\n\n") if p.strip()]
            row["status"] = "skipped"
            row["paragraphs"] = len(paras)
            row["words"] = len(content.split())
            return row
        except Exception:
            pass

    t0 = time.time()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        # 1. Extract genuine narrative candidate sentences
        doc = extract_report_sentences(data, file_path=path)

        # 2. Select extractive sentences to form paragraphs
        summary_text = ""
        info: Dict[str, Any] = {}

        if method in ("gemma", "qwen", "nemotron") and client is not None:
            try:
                summary_text, selected, info = llm_extractive_paragraphs(client, doc, target_words=target_words)
            except Exception as e:
                logger.warning("%s generation failed on %s (%s); falling back to heuristic.", method, stem, e)
                summary_text, selected = heuristic_paragraph_summary(doc, max_words=target_words)
                info = {"n_paragraphs": len(summary_text.split("\n\n")), "fallback": True}
        else:
            summary_text, selected = heuristic_paragraph_summary(doc, max_words=target_words)
            info = {"n_paragraphs": len(summary_text.split("\n\n")), "method": "baseline"}

        summary_text = summary_text.strip()
        if not summary_text:
            raise ValueError(f"No summary text could be generated for {path}.")

        # 3. Strict Verbatim Substring Check
        paras = [p for p in summary_text.split("\n\n") if p.strip()]
        if selected and doc.source_text:
            verify_verbatim_summary(selected, doc.source_text)

        # 4. Save clean text outputs (.txt)
        os.makedirs(os.path.join(out_dir, stem), exist_ok=True)

        with open(out_file_root, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")

        with open(out_file_sub, "w", encoding="utf-8") as f:
            f.write(summary_text + "\n")

        # Clean up any obsolete JSON artifacts from previous runs
        obsolete_json = os.path.join(out_dir, stem, "result.json")
        if os.path.exists(obsolete_json):
            try:
                os.remove(obsolete_json)
            except OSError:
                pass

        elapsed = round(time.time() - t0, 2)
        row["status"] = "ok"
        row["paragraphs"] = len(paras)
        row["words"] = len(summary_text.split())
        row["time_s"] = elapsed

    except Exception as e:
        row["status"] = "error"
        row["error"] = f"{type(e).__name__}: {e}"
        row["time_s"] = round(time.time() - t0, 2)
        # Log error
        err_dir = os.path.join(out_dir, stem)
        os.makedirs(err_dir, exist_ok=True)
        with open(os.path.join(err_dir, "error.txt"), "w", encoding="utf-8") as ef:
            ef.write(traceback.format_exc())

    return row


def inspect_file(path: str) -> None:
    """Inspect extracted sentences and candidate sections for one report JSON."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    doc = extract_report_sentences(data, file_path=path)

    print(f"\n=======================================================")
    print(f"File: {path}")
    print(f"Title: {doc.title}")
    print(f"Overview sentences: {len(doc.overview_sentences)}")
    print(f"Finding sentences:  {len(doc.finding_sentences)}")
    print(f"Rec sentences:      {len(doc.recommendation_sentences)}")
    print(f"Total source words: {doc.total_words}")
    print(f"=======================================================\n")

    if doc.overview_sentences:
        print("[Section A: Overview (first 3)]")
        for s in doc.overview_sentences[:3]:
            print(f"  [{s.uid}] {s.text}")

    if doc.finding_sentences:
        print("\n[Section B: Key Findings (top 4)]")
        sorted_f = sorted(doc.finding_sentences, key=lambda s: s.severity_score, reverse=True)
        for s in sorted_f[:4]:
            print(f"  [{s.uid}] ({s.severity} | {s.finding_title}) {s.text}")

    if doc.recommendation_sentences:
        print("\n[Section C: Recommendations (top 3)]")
        sorted_r = sorted(doc.recommendation_sentences, key=lambda s: s.severity_score, reverse=True)
        for s in sorted_r[:3]:
            print(f"  [{s.uid}] ({s.finding_title}) {s.text}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extractive paragraph summarization of audit-report JSON files"
    )
    ap.add_argument("--file", help="Path to a single JSON file to summarize")
    ap.add_argument("--in_dir", default=DEFAULT_IN_DIR, help="Input directory of JSON reports")
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR, help="Output folder for text summaries")
    ap.add_argument("--limit", type=int, default=None, help="Process only the first N files")
    ap.add_argument("--force", action="store_true", help="Recompute summaries that already exist")
    ap.add_argument(
        "--method",
        choices=["gemma", "qwen", "nemotron", "baseline"],
        default="gemma",
        help="Summarization method: 'gemma' (local 31B), 'qwen' (Qwen-2.5-32B), 'nemotron' (Nemotron), or 'baseline' (fast heuristic)",
    )
    ap.add_argument(
        "--model_path",
        default=DEFAULT_MODEL_PATH,
        help="Folder containing local Gemma weights",
    )
    ap.add_argument(
        "--qwen_name",
        default="qwen2.5:32b",
        help="Ollama model name for Qwen (default: qwen2.5:32b)",
    )
    ap.add_argument(
        "--nemotron_name",
        default="nemotron",
        help="Ollama model name for Nemotron (default: nemotron)",
    )
    ap.add_argument(
        "--target_words",
        type=int,
        default=250,
        help="Soft target word count for the summary (default 250)",
    )
    ap.add_argument("--inspect", metavar="JSON_FILE", help="Inspect sentence extraction for one file")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.inspect:
        inspect_file(args.inspect)
        return

    # Determine files to process
    if args.file:
        files = [os.path.abspath(args.file)]
        if not os.path.isfile(files[0]):
            raise SystemExit(f"Input file not found: {args.file}")
    else:
        files = sorted(
            glob.glob(os.path.join(args.in_dir, "*.json")),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
            if os.path.splitext(os.path.basename(p))[0].isdigit()
            else p,
        )
        if args.limit:
            files = files[: args.limit]
        if not files:
            raise SystemExit(f"No JSON files found in {args.in_dir}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Initialize client if using an LLM
    client: Any = None
    if args.method == "gemma":
        print(f"Loading Gemma model from: {args.model_path} ...", flush=True)
        t_load = time.time()
        try:
            client = GemmaClient(model_path=args.model_path)
            print(f"Gemma loaded successfully in {time.time() - t_load:.1f}s.\n", flush=True)
        except GemmaError as e:
            print(f"Warning: Could not load Gemma: {e}\nFalling back to baseline method.", flush=True)
            args.method = "baseline"
    elif args.method == "qwen":
        print(f"Connecting to Qwen ({args.qwen_name}) ...", flush=True)
        try:
            client = OllamaClient(model_name=args.qwen_name)
            client.ping()
            print(f"Connected to Qwen successfully.\n", flush=True)
        except Exception as e:
            print(f"Warning: Could not connect to Qwen: {e}\nFalling back to baseline method.", flush=True)
            args.method = "baseline"
    elif args.method == "nemotron":
        print(f"Connecting to Nemotron ({args.nemotron_name}) ...", flush=True)
        try:
            client = OllamaClient(model_name=args.nemotron_name)
            client.ping()
            print(f"Connected to Nemotron successfully.\n", flush=True)
        except Exception as e:
            print(f"Warning: Could not connect to Nemotron: {e}\nFalling back to baseline method.", flush=True)
            args.method = "baseline"

    total = len(files)
    print(f"Extractive Paragraph Summarizer")
    print(f"Method: {args.method}  Files: {total}  Target words: {args.target_words}")
    print(f"Output directory: {args.out_dir}\n")
    print(f"{'#':<4} {'File':<14} {'Status':<8} {'Paras':<6} {'Words':<8} {'Time (s)':<10}")
    print("-" * 55)

    rows: List[Dict[str, Any]] = []
    for idx, fpath in enumerate(files, start=1):
        row = process_file(
            fpath,
            args.out_dir,
            client=client,
            method=args.method,
            target_words=args.target_words,
            force=args.force,
        )
        rows.append(row)
        time_str = f"{row['time_s']:.1f}s" if row["time_s"] else "-"
        detail = f"{row['error'][:80]}" if row["error"] else ""
        print(
            f"[{idx:>2}/{total}] {row['file']:<14} {row['status']:<8} "
            f"{row['paragraphs']:<6} {row['words']:<8} {time_str:<10} {detail}",
            flush=True,
        )

    # Clean summary output
    ok_count = sum(1 for r in rows if r["status"] == "ok")
    skipped_count = sum(1 for r in rows if r["status"] == "skipped")
    err_count = sum(1 for r in rows if r["status"] == "error")
    avg_words = (
        sum(r["words"] for r in rows if r["status"] in ("ok", "skipped"))
        / max(ok_count + skipped_count, 1)
    )

    print("\n" + "=" * 55)
    print(
        f"Completed: {total} files | OK: {ok_count} | Skipped: {skipped_count} | Errors: {err_count}"
    )
    print(f"Average summary length: {avg_words:.1f} words")
    print(f"Outputs written to: {args.out_dir}/<file>.txt and {args.out_dir}/<file>/summary.txt")
    print("=" * 55)

    # Write summary log
    summary_log = os.path.join(args.out_dir, "run_summary.txt")
    with open(summary_log, "w", encoding="utf-8") as sf:
        sf.write(f"Extractive Summarization Run Report\n")
        sf.write(f"Method: {args.method}  Total files: {total}\n")
        sf.write(f"OK: {ok_count}  Skipped: {skipped_count}  Errors: {err_count}\n")
        sf.write(f"Average length: {avg_words:.1f} words\n\n")
        sf.write(f"{'File':<16} {'Status':<10} {'Paras':<8} {'Words':<8} {'Time (s)':<10}\n")
        sf.write("-" * 55 + "\n")
        for r in rows:
            sf.write(
                f"{r['file']:<16} {r['status']:<10} {r['paragraphs']:<8} {r['words']:<8} {r['time_s']:<10.2f}\n"
            )


if __name__ == "__main__":
    main()