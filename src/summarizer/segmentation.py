"""
Sentence and unit segmentation for security audit reports.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from summarizer.structure import (
    classify_section,
    detect_heading,
    extract_finding_id,
    is_table_line,
    severity_score,
)

_ABBREV_RE = re.compile(r"\b(e\.g|i\.e|etc|vs|approx|fig|sec|cf|dr|mr|mrs|ms|inc|ltd|pvt|co|corp|dept|gov|no|vol|st|al|prof)\.", re.I)
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


@dataclass
class Unit:
    """A granular extractive unit (prose sentence or table row)."""

    uid: int
    text: str
    page: int
    section_index: int
    section_title: str
    section_type: str
    finding_id: Optional[str]
    is_table_row: bool
    words: int
    severity: float = 0.0
    eff_severity: float = 0.0
    position: float = 0.0
    section_pos: float = 0.0


@dataclass
class Document:
    """Parsed document containing extracted units and full text."""

    name: str
    units: List[Unit]
    source_text: str
    emb: Optional[np.ndarray] = None
    sim: Optional[np.ndarray] = None
    X: Optional[np.ndarray] = None

    @property
    def total_words(self) -> int:
        """Total word count of all units in the document."""
        return sum(u.words for u in self.units)


def split_sentences(text: str) -> List[str]:
    """Split prose text into sentences with abbreviation and decimal protection.

    Args:
        text: Input paragraph or text block.

    Returns:
        List of sentence strings.
    """
    protected = _ABBREV_RE.sub(lambda m: m.group(0).replace(".", "<DOT>"), text)
    protected = re.sub(r"(?<=\d)\.(?=\d)", "<DOT>", protected)
    parts = _SPLIT_RE.split(protected)
    return [p.replace("<DOT>", ".").strip() for p in parts if p and p.strip()]


def build_units(
    cleaned_pages: List[Dict[str, Any]],
    min_words_unit: int = 5,
    min_words_table_unit: int = 3,
) -> Tuple[List[Unit], str]:
    """Segment cleaned pages into discrete units and construct canonical source text.

    Args:
        cleaned_pages: Output from clean_pages.
        min_words_unit: Minimum words for prose sentences (unless severity or finding ID).
        min_words_table_unit: Minimum words for table rows.

    Returns:
        Tuple of (list of Unit objects, source_text string for verbatim checks).
    """
    units: List[Unit] = []
    state = {"sec_idx": -1, "title": "Preamble", "type": "other", "fid": None}
    heading_sev: Dict[str, float] = {}
    buf: List[str] = []

    def emit(text: str, page: int, is_table: bool) -> None:
        text = re.sub(r"\s+", " ", text).strip()
        words = len(text.split())
        sev = severity_score(text)
        own_fid = extract_finding_id(text)
        min_w = min_words_table_unit if is_table else min_words_unit

        # Drop fragments shorter than min_w, except units that contain a severity term or finding ID
        if words < 2 or (words < min_w and sev == 0.0 and own_fid is None):
            return

        units.append(
            Unit(
                uid=len(units),
                text=text,
                page=page,
                section_index=state["sec_idx"],
                section_title=state["title"],
                section_type=state["type"],
                finding_id=state["fid"] or own_fid,
                is_table_row=is_table,
                words=words,
                severity=sev,
            )
        )

    def flush(page: int) -> None:
        if not buf:
            return
        para = re.sub(r"\s+", " ", " ".join(buf))
        buf.clear()
        for s in split_sentences(para):
            emit(s, page, False)

    for p in cleaned_pages:
        page = p["page"]
        for line in p["lines"]:
            if not line:
                flush(page)
                continue

            title = detect_heading(line)
            if title:
                flush(page)
                state["sec_idx"] += 1
                state["title"] = title
                state["type"] = classify_section(title)
                fid = extract_finding_id(title)
                state["fid"] = fid
                if fid:
                    heading_sev[fid] = max(heading_sev.get(fid, 0.0), severity_score(title))
                continue

            if is_table_line(line):
                flush(page)
                emit(line, page, True)
                continue

            buf.append(line)
        flush(page)

    # Compute positions
    n = len(units)
    sec_counts = Counter(u.section_index for u in units)
    sec_seen: Counter[int] = Counter()
    for u in units:
        u.position = u.uid / max(n - 1, 1)
        u.section_pos = sec_seen[u.section_index] / max(sec_counts[u.section_index] - 1, 1)
        sec_seen[u.section_index] += 1

    # Propagate each finding's maximum severity across its group
    group_max = dict(heading_sev)
    for u in units:
        if u.finding_id:
            group_max[u.finding_id] = max(group_max.get(u.finding_id, 0.0), u.severity)

    for u in units:
        u.eff_severity = (
            max(u.severity, group_max.get(u.finding_id, 0.0))
            if u.finding_id
            else u.severity
        )

    # Reconstruct document source text for exact verbatim substring sanity check
    # Every emitted unit.text was formed from the text of these pages
    source_text = " \n ".join(u.text for u in units)

    return units, source_text
