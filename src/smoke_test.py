#!/usr/bin/env python3
"""
Quick smoke test for JSON extractive paragraph summarization.
Runs instantly (using the deterministic heuristic summarizer).
"""

import json
import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from summarizer.json_report import extract_report_sentences
from summarizer.selection import heuristic_paragraph_summary, verify_verbatim_summary

def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SRC_DIR, "..", "data2", "1.json")
    print(f"Testing extractive summarizer on: {json_path}")
    
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"File not found: {json_path}")
        
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
        
    doc = extract_report_sentences(data, file_path=json_path)
    print(f"Report: {doc.title}")
    print(f"Overview sentences: {len(doc.overview_sentences)}")
    print(f"Finding sentences:  {len(doc.finding_sentences)}")
    print(f"Rec sentences:      {len(doc.recommendation_sentences)}")
    
    summary_text, selected = heuristic_paragraph_summary(doc, max_words=250)
    paras = [p for p in summary_text.split("\n\n") if p.strip()]
    
    print(f"\nGenerated Summary ({len(paras)} paragraphs, {len(summary_text.split())} words):\n")
    print("-" * 60)
    print(summary_text)
    print("-" * 60)
    
    # Verbatim check
    verify_verbatim_summary(selected, doc.source_text)
    print("\nSUCCESS: All sentences verified 100% verbatim from source report text!")

if __name__ == "__main__":
    main()