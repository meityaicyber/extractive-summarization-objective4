"""
Extractive summarization of security audit reports.

Heavy dependencies (torch, transformers) are not imported here. Import
summarizer.model directly when the transformer scorer is needed.
"""

from summarizer.extraction import extract_pages, clean_pages
from summarizer.structure import (
    SECTION_TYPES,
    SECTION_KEYWORDS,
    detect_heading,
    classify_section,
    severity_score,
    severity_to_level,
    extract_finding_id,
    is_table_line,
)
from summarizer.segmentation import Unit, Document, split_sentences, build_units
from summarizer.labels import silver_labels, weak_labels, unigram_f1
from summarizer.selection import select_units, word_budget, verify_verbatim_summary
from summarizer.evaluation import evaluate_summary, silver_auc

__all__ = [
    "extract_pages",
    "clean_pages",
    "SECTION_TYPES",
    "SECTION_KEYWORDS",
    "detect_heading",
    "classify_section",
    "severity_score",
    "severity_to_level",
    "extract_finding_id",
    "is_table_line",
    "Unit",
    "Document",
    "split_sentences",
    "build_units",
    "silver_labels",
    "weak_labels",
    "unigram_f1",
    "select_units",
    "word_budget",
    "verify_verbatim_summary",
    "evaluate_summary",
    "silver_auc",
]