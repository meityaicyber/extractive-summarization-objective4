"""
Selection algorithm with coverage, anchor, and MMR redundancy control rules.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from summarizer.segmentation import Unit

logger = logging.getLogger(__name__)


def word_budget(
    total_words: int,
    budget_ratio: float = 0.12,
    min_budget_words: int = 40,
) -> int:
    """Compute target word budget for summary.

    Args:
        total_words: Total word count in source document.
        budget_ratio: Share of source words (default 0.12 = 12%).
        min_budget_words: Minimum floor for word budget (default 40).

    Returns:
        Integer word count budget.
    """
    return max(min_budget_words, int(total_words * budget_ratio))


def verify_verbatim_summary(units: Sequence[Any], source_text: str) -> None:
    """Sanity check that fails if any summary sentence is not a verbatim substring of source text.

    Args:
        units: Selected summary units or sentences (or strings).
        source_text: Cleaned full text of source document.

    Raises:
        AssertionError: If any unit text is not a verbatim substring.
    """
    norm_source = re.sub(r"\s+", " ", source_text).strip()
    for u in units:
        text = u.text if hasattr(u, "text") else str(u)
        norm_unit = re.sub(r"\s+", " ", text).strip()
        if norm_unit not in norm_source:
            uid_str = getattr(u, "uid", "?")
            raise AssertionError(
                f"Sanity check failed: selected unit {uid_str} is not a verbatim substring of source text:\n'{text}'"
            )


def heuristic_paragraph_summary(
    doc: Any,
    max_words: int = 300,
) -> Tuple[str, List[Any]]:
    """Generate a 2-3 paragraph extractive summary using deterministic coverage rules.

    Paragraph 1: Overview & Posture (up to ~75 words)
    Paragraph 2: Key Findings & Impacts (up to ~140 words, prioritizing Critical/High/Medium)
    Paragraph 3: Recommendations (up to ~85 words)

    Returns:
        (summary_text, list of selected sentence objects)
    """
    p1_sents: List[Any] = []
    p2_sents: List[Any] = []
    p3_sents: List[Any] = []

    # 1. Paragraph 1: Overview
    w1 = 0
    for s in doc.overview_sentences:
        if w1 + s.words <= 85 or not p1_sents:
            p1_sents.append(s)
            w1 += s.words
        if w1 >= 75:
            break

    # 2. Paragraph 2: Findings & Impacts
    # Group finding sentences by finding title or id, sort by severity
    sorted_findings = sorted(
        doc.finding_sentences,
        key=lambda s: (s.severity_score, len(s.text)),
        reverse=True,
    )
    w2 = 0
    seen_fids = set()
    for s in sorted_findings:
        fid = s.finding_id or s.finding_title
        # Prioritize covering distinct findings first
        if fid not in seen_fids or len(seen_fids) < 3:
            if w2 + s.words <= 150 or not p2_sents:
                p2_sents.append(s)
                w2 += s.words
                if fid:
                    seen_fids.add(fid)
        if w2 >= 130:
            break

    # 3. Paragraph 3: Recommendations
    sorted_recs = sorted(
        doc.recommendation_sentences,
        key=lambda s: (s.severity_score, len(s.text)),
        reverse=True,
    )
    w3 = 0
    seen_rec_fids = set()
    for s in sorted_recs:
        fid = s.finding_id or s.finding_title
        if fid not in seen_rec_fids or len(seen_rec_fids) < 3:
            if w3 + s.words <= 95 or not p3_sents:
                p3_sents.append(s)
                w3 += s.words
                if fid:
                    seen_rec_fids.add(fid)
        if w3 >= 80:
            break

    # Sort sentences within each paragraph by original document order (uid)
    p1_sents.sort(key=lambda s: s.uid)
    p2_sents.sort(key=lambda s: s.uid)
    p3_sents.sort(key=lambda s: s.uid)

    def _term(t: str) -> str:
        t = t.strip()
        return t if (not t or t[-1] in ".!?;:") else t + "."

    # Assemble paragraphs
    paras: List[str] = []
    if p1_sents:
        paras.append(" ".join(_term(s.text) for s in p1_sents))
    if p2_sents:
        paras.append(" ".join(_term(s.text) for s in p2_sents))
    if p3_sents:
        paras.append(" ".join(_term(s.text) for s in p3_sents))

    summary_text = "\n\n".join(paras)
    selected_all = p1_sents + p2_sents + p3_sents
    return summary_text, selected_all


def select_units(
    units: Sequence[Unit],
    scores: np.ndarray,
    sim_matrix: np.ndarray,
    total_words: int,
    budget_ratio: float = 0.12,
    high_severity_threshold: float = 0.8,
    mmr_lambda: float = 0.7,
    known_high_severity_fids: Optional[Sequence[str]] = None,
    source_text: Optional[str] = None,
) -> Tuple[List[int], Dict[int, str], Dict[str, Any]]:
    """Select extractive summary units according to coverage, anchor, and MMR rules.

    Rules in order:
      1. Coverage: For each finding ID whose effective severity is at least 0.8,
         include the highest-scoring unit from that finding, and fail loudly if a
         high-severity finding has no units.
      2. Anchor: Include highest-scoring unit from exec_summary, recommendations,
         and conclusion when they exist.
      3. Greedy fill: Fill remaining word budget greedily by score, with Maximal
         Marginal Relevance using sentence-embedding cosine similarity to reduce
         redundancy (lambda 0.7).
      4. Output units in original document order.
      5. If mandatory units exceed budget, keep them and print a warning.

    Args:
        units: Sequence of Unit objects.
        scores: 1D array of relevance probabilities for each unit.
        sim_matrix: (N, N) cosine similarity matrix between units.
        total_words: Total word count of document.
        budget_ratio: Share of source words (default 0.12).
        high_severity_threshold: Threshold for high-severity findings (default 0.8).
        mmr_lambda: Weight for relevance vs redundancy in MMR (default 0.7).
        known_high_severity_fids: Optional list of expected high severity finding IDs.
        source_text: Optional source text for verbatim verification.

    Returns:
        Tuple of:
          - chosen_indices: list of unit indices sorted by document order
          - reasons: mapping of unit index to selection reason string
          - info: dict with budget, used words, mandatory words, over_budget flag
    """
    n = len(units)
    budget = word_budget(total_words, budget_ratio)
    if n == 0:
        return [], {}, {"budget": budget, "used": 0, "mandatory_words": 0, "over_budget": False}

    scores = np.asarray(scores, dtype=float)
    chosen: List[int] = []
    reasons: Dict[int, str] = {}
    used = 0

    def take(idx: int, reason: str) -> None:
        nonlocal used
        if idx in reasons:
            return
        chosen.append(idx)
        reasons[idx] = reason
        used += units[idx].words

    # --- Rule 1: Coverage Rule ---
    # Find all finding IDs in document that have effective severity >= high_severity_threshold
    doc_finding_units: Dict[str, List[int]] = {}
    for i, u in enumerate(units):
        if u.finding_id and u.eff_severity >= high_severity_threshold:
            doc_finding_units.setdefault(u.finding_id, []).append(i)

    # Check for known high severity findings that must be covered
    all_target_fids = set(doc_finding_units.keys())
    if known_high_severity_fids:
        for fid in known_high_severity_fids:
            if fid not in doc_finding_units or not doc_finding_units[fid]:
                raise RuntimeError(
                    f"Coverage rule violation: high-severity finding '{fid}' has no units extracted from document!"
                )
            all_target_fids.add(fid)

    for fid in sorted(all_target_fids):
        f_units = doc_finding_units.get(fid, [])
        if not f_units:
            raise RuntimeError(
                f"Coverage rule violation: high-severity finding '{fid}' has no extracted units!"
            )
        best_u = max(f_units, key=lambda i: scores[i])
        take(best_u, f"coverage:{fid}")

    # --- Rule 2: Anchor Rule ---
    # Include highest-scoring unit from exec_summary, recommendations, conclusion when they exist
    for stype in ("exec_summary", "recommendations", "conclusion"):
        idxs = [i for i, u in enumerate(units) if u.section_type == stype]
        if idxs:
            best_idx = max(idxs, key=lambda i: scores[i])
            take(best_idx, f"anchor:{stype}")

    mandatory_words = used
    over_budget = mandatory_words > budget
    if over_budget:
        msg = (
            f"[WARNING] Mandatory units ({mandatory_words} words) exceed word budget "
            f"({budget} words); keeping all mandatory units."
        )
        logger.warning(msg)
        print(msg)

    # --- Rule 3: Greedy Fill with MMR ---
    if not over_budget:
        remaining = [i for i in range(n) if i not in reasons]
        while remaining:
            rem = np.asarray(remaining, dtype=int)
            if chosen and sim_matrix.size > 0:
                redundancy = sim_matrix[np.ix_(rem, np.asarray(chosen, dtype=int))].max(axis=1)
            else:
                redundancy = np.zeros(len(rem))

            mmr = mmr_lambda * scores[rem] - (1.0 - mmr_lambda) * redundancy

            # Greedily pick the highest-scoring candidate that fits within budget
            picked = None
            for k in np.argsort(-mmr):
                cand = remaining[k]
                if used + units[cand].words <= budget:
                    picked = cand
                    break

            if picked is None:
                break

            take(picked, "mmr")
            remaining.remove(picked)

    # Output units in original document order
    chosen_sorted = sorted(chosen)

    # --- Sanity Check: Verbatim Substring ---
    if source_text is not None:
        selected_units = [units[i] for i in chosen_sorted]
        verify_verbatim_summary(selected_units, source_text)

    info = {
        "budget": budget,
        "used": used,
        "mandatory_words": mandatory_words,
        "over_budget": over_budget,
    }
    return chosen_sorted, reasons, info
