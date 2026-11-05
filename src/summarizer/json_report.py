"""
Extraction module for audit-report JSON schemas in data2/.

Extracts genuine narrative prose sentences from report JSON and categorizes
them into thematic roles for extractive paragraph summarization:
  - 'overview': executive summary narrative, scope, and audit posture
  - 'findings': observation descriptions, security impacts, root causes
  - 'recommendations': remediation guidance and corrective action steps

Ensures 100% verbatim traceability against original source text.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from summarizer.segmentation import split_sentences
from summarizer.structure import severity_score

# Keys marking an audit report in this schema
REPORT_MARKER_KEYS: Tuple[str, ...] = (
    "detailed_observations",
    "executive_summary",
    "engagement_scope",
    "report_meta",
)

SEVERITY_WEIGHTS = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "moderate": 0.5,
    "low": 0.25,
    "informational": 0.1,
    "info": 0.1,
    "none": 0.0,
}


@dataclass
class ExtractiveSentence:
    """A verbatim sentence extracted from an audit report."""

    uid: int
    text: str
    role: str  # 'overview', 'findings', 'recommendations'
    section: str
    finding_id: Optional[str] = None
    finding_title: Optional[str] = None
    severity: str = "none"
    severity_score: float = 0.0
    status: Optional[str] = None
    words: int = 0

    def __post_init__(self):
        if not self.words:
            self.words = len(self.text.split())


@dataclass
class ReportDocument:
    """Document containing parsed extractive candidate sentences and raw source text."""

    file_path: str
    stem: str
    title: str
    overview_sentences: List[ExtractiveSentence] = field(default_factory=list)
    finding_sentences: List[ExtractiveSentence] = field(default_factory=list)
    recommendation_sentences: List[ExtractiveSentence] = field(default_factory=list)
    all_sentences: List[ExtractiveSentence] = field(default_factory=list)
    source_text: str = ""
    total_words: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


def is_audit_report(data: Any) -> bool:
    """Return True if the parsed JSON uses this audit report schema."""
    return isinstance(data, dict) and any(k in data for k in REPORT_MARKER_KEYS)


def _clean_text(s: str) -> str:
    """Normalize whitespace and strip unneeded wrapping quotes."""
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip("\"' ")


def _parse_severity(sev_val: Any) -> Tuple[str, float]:
    """Parse severity string and return (canonical_name, numeric_weight)."""
    if not isinstance(sev_val, str):
        return ("none", 0.0)
    name = sev_val.strip().lower()
    for k, w in SEVERITY_WEIGHTS.items():
        if k in name:
            return (k.capitalize(), w)
    return (sev_val.strip() or "none", severity_score(sev_val))


def extract_report_sentences(data: Dict[str, Any], file_path: str = "") -> ReportDocument:
    """Extract narrative sentences from audit-report JSON for paragraph summarization.

    Args:
        data: Parsed report JSON.
        file_path: Optional source file path.

    Returns:
        ReportDocument populated with verbatim candidate sentences and source text.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0] if file_path else "doc"
    meta = data.get("report_meta", {}) if isinstance(data.get("report_meta"), dict) else {}
    title = meta.get("report_title") or meta.get("report_id") or f"Audit Report {stem}"

    overview_sentences: List[ExtractiveSentence] = []
    finding_sentences: List[ExtractiveSentence] = []
    recommendation_sentences: List[ExtractiveSentence] = []
    all_sentences: List[ExtractiveSentence] = []

    source_chunks: List[str] = []
    uid_counter = 0

    def add_sentence(
        text: str,
        role: str,
        section: str,
        finding_id: Optional[str] = None,
        finding_title: Optional[str] = None,
        severity: str = "none",
        severity_score: float = 0.0,
        status: Optional[str] = None,
    ) -> Optional[ExtractiveSentence]:
        nonlocal uid_counter
        t = _clean_text(text)
        words = len(t.split())
        if words < 3:
            return None
        # Exclude internal technical artifacts or image placeholders
        if "[IMAGE_" in t or "[FIGURE_" in t or "[TABLE_" in t:
            return None

        uid_counter += 1
        sent = ExtractiveSentence(
            uid=uid_counter,
            text=t,
            role=role,
            section=section,
            finding_id=finding_id,
            finding_title=finding_title,
            severity=severity,
            severity_score=severity_score,
            status=status,
            words=words,
        )
        all_sentences.append(sent)
        if role == "overview":
            overview_sentences.append(sent)
        elif role == "findings":
            finding_sentences.append(sent)
        elif role == "recommendations":
            recommendation_sentences.append(sent)
        return sent

    # -------------------------------------------------------------------------
    # 1. Executive Summary & Overview
    # -------------------------------------------------------------------------
    es = data.get("executive_summary")
    if isinstance(es, dict):
        narrative = es.get("narrative")
        if isinstance(narrative, str) and narrative.strip():
            source_chunks.append(narrative)
            for s in split_sentences(narrative):
                add_sentence(s, role="overview", section="Executive Summary")

    # If narrative is empty or short, harvest scope descriptions
    scope = data.get("engagement_scope")
    if isinstance(scope, dict):
        assets = scope.get("assets")
        if isinstance(assets, list):
            for asset in assets:
                if isinstance(asset, dict):
                    desc = asset.get("asset_description")
                    if isinstance(desc, str) and desc.strip():
                        source_chunks.append(desc)
                        if len(overview_sentences) < 2:
                            for s in split_sentences(desc):
                                add_sentence(s, role="overview", section="Scope")

    # -------------------------------------------------------------------------
    # 2. Detailed Observations: Findings & Recommendations
    # -------------------------------------------------------------------------
    findings_list = data.get("detailed_observations")
    if isinstance(findings_list, list):
        for item in findings_list:
            if not isinstance(item, dict):
                continue

            fid = item.get("finding_id") or ""
            ftitle = item.get("title") or ""
            fsev, fscore = _parse_severity(item.get("severity"))
            fstatus = item.get("status")

            # Harvest observation description
            desc = item.get("observation_description")
            if isinstance(desc, str) and desc.strip():
                source_chunks.append(desc)
                for s in split_sentences(desc):
                    add_sentence(
                        s,
                        role="findings",
                        section="Findings",
                        finding_id=fid,
                        finding_title=ftitle,
                        severity=fsev,
                        severity_score=fscore,
                        status=fstatus,
                    )

            # Harvest security impact
            impact = item.get("impact")
            if isinstance(impact, str) and impact.strip():
                source_chunks.append(impact)
                for s in split_sentences(impact):
                    add_sentence(
                        s,
                        role="findings",
                        section="Impact",
                        finding_id=fid,
                        finding_title=ftitle,
                        severity=fsev,
                        severity_score=fscore,
                        status=fstatus,
                    )

            # Harvest root cause
            root = item.get("likely_root_cause")
            if isinstance(root, str) and root.strip():
                source_chunks.append(root)
                for s in split_sentences(root):
                    add_sentence(
                        s,
                        role="findings",
                        section="Root Cause",
                        finding_id=fid,
                        finding_title=ftitle,
                        severity=fsev,
                        severity_score=fscore,
                        status=fstatus,
                    )

            # Harvest recommendations
            rec = item.get("recommendation")
            if isinstance(rec, str) and rec.strip():
                source_chunks.append(rec)
                for s in split_sentences(rec):
                    add_sentence(
                        s,
                        role="recommendations",
                        section="Recommendations",
                        finding_id=fid,
                        finding_title=ftitle,
                        severity=fsev,
                        severity_score=fscore,
                        status=fstatus,
                    )

    # -------------------------------------------------------------------------
    # 3. Appendices: Criteria as context fallback if needed
    # -------------------------------------------------------------------------
    appendices = data.get("appendices")
    if isinstance(appendices, dict):
        criteria = appendices.get("risk_rating_criteria")
        if isinstance(criteria, list):
            for crit in criteria:
                if isinstance(crit, dict):
                    cdesc = crit.get("description")
                    if isinstance(cdesc, str) and cdesc.strip():
                        source_chunks.append(cdesc)
                        if not finding_sentences:
                            crating = crit.get("rating", "")
                            for s in split_sentences(cdesc):
                                add_sentence(
                                    s,
                                    role="findings",
                                    section="Risk Criteria",
                                    finding_title=crating,
                                    severity=crating,
                                    severity_score=severity_score(crating),
                                )

    # Combine full text for exact verbatim substring validation
    source_text = " \n ".join(source_chunks)
    total_words = sum(s.words for s in all_sentences)

    return ReportDocument(
        file_path=file_path,
        stem=stem,
        title=title,
        overview_sentences=overview_sentences,
        finding_sentences=finding_sentences,
        recommendation_sentences=recommendation_sentences,
        all_sentences=all_sentences,
        source_text=source_text,
        total_words=total_words,
        metadata={
            "audit_type": meta.get("audit_type"),
            "report_type": meta.get("report_type"),
            "audit_period": meta.get("audit_period"),
            "n_overview": len(overview_sentences),
            "n_findings": len(finding_sentences),
            "n_recs": len(recommendation_sentences),
        },
    )