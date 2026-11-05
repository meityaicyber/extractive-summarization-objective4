"""
Silver and weak label generation for sentence relevance.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List, Sequence

import numpy as np

from summarizer.segmentation import Unit, split_sentences

STOPWORDS = set(
    "a an the of and or to in on for with by is are was were be been this that "
    "these those as at from it its into than then".split()
)


def content_tokens(text: str) -> List[str]:
    """Tokenize text into lowercase alphanumeric content tokens excluding stopwords.

    Args:
        text: Input string.

    Returns:
        List of content tokens.
    """
    return [
        t for t in re.findall(r"[a-z0-9]+", text.lower())
        if t not in STOPWORDS and len(t) > 1
    ]


def unigram_f1(a: List[str], b: List[str]) -> float:
    """Compute token-level unigram F1 overlap between two lists of tokens.

    Args:
        a: First token list.
        b: Second token list.

    Returns:
        F1 score in [0.0, 1.0].
    """
    if not a or not b:
        return 0.0

    overlap = sum((Counter(a) & Counter(b)).values())
    if overlap == 0:
        return 0.0

    p = overlap / len(a)
    r = overlap / len(b)
    return 2 * p * r / (p + r)


def silver_labels(
    units: Sequence[Unit],
    reference_text: str,
    threshold: float = 0.4,
) -> np.ndarray:
    """Generate binary silver labels based on unigram F1 overlap with reference summary.

    A unit is labeled 1 if its unigram F1 with any sentence in the reference summary
    is at least threshold (default 0.4), otherwise 0.

    Args:
        units: Sequence of Unit objects.
        reference_text: Reference summary text.
        threshold: Unigram F1 threshold (default 0.4).

    Returns:
        1D numpy array of binary integers (0 or 1).
    """
    ref_sents = [s for s in split_sentences(reference_text) if len(s.split()) >= 4]
    ref_tokens = [content_tokens(s) for s in ref_sents]

    y = np.zeros(len(units), dtype=int)
    for i, u in enumerate(units):
        u_tok = content_tokens(u.text)
        best_f1 = max((unigram_f1(u_tok, rt) for rt in ref_tokens), default=0.0)
        y[i] = int(best_f1 >= threshold)

    return y


def weak_labels(
    units: Sequence[Unit],
    high_severity_threshold: float = 0.8,
) -> np.ndarray:
    """Generate heuristic weak labels when no reference summary is available.

    Args:
        units: Sequence of Unit objects.
        high_severity_threshold: Effective severity threshold for positive label.

    Returns:
        1D numpy array of binary integers.
    """
    y = [
        1 if (
            u.eff_severity >= high_severity_threshold
            or u.section_type in ("exec_summary", "conclusion")
        ) else 0
        for u in units
    ]
    return np.asarray(y, dtype=int)
