"""
Document structure recovery, heading detection, section classification, and severity parsing.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

SECTION_TYPES = [
    "exec_summary",
    "scope",
    "methodology",
    "findings",
    "recommendations",
    "conclusion",
    "appendix",
    "other",
]

# Order matters: more specific keywords are checked first.
SECTION_KEYWORDS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("exec_summary", ("executive summary", "management summary")),
    ("recommendations", ("recommendation", "remediation", "action plan", "mitigation", "corrective")),
    ("conclusion", ("conclusion", "overall assessment", "verdict", "opinion")),
    ("findings", ("finding", "observation", "vulnerabilit", "non-conform", "nonconform", "issue", "weakness", "gap")),
    ("methodology", ("methodolog", "approach", "tools used", "test approach", "testing approach")),
    ("scope", ("scope", "objective", "background", "introduction", "purpose")),
    ("appendix", ("annex", "appendix", "evidence", "glossary")),
    ("exec_summary", ("summary", "overview", "highlights")),
)

NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+([A-Z][A-Za-z0-9 ,&/()\-:]{2,80})$")
KEYWORD_HEADING = re.compile(
    r"^(executive summary|summary|overview|scope|objectives?|background|introduction|"
    r"methodology|approach|findings?|observations|vulnerabilities|recommendations?|"
    r"remediation|conclusions?|opinion|annex(?:ure)?|appendix|evidence)\b",
    re.I,
)

FID_WORD_RE = re.compile(r"\b(?:Finding|Issue)\s*#?\s*(\d{1,4})\b", re.I)
FID_CODE_RE = re.compile(r"\b(?:F|NC|VULN)-(\d{1,4})\b", re.I)
FINDING_LINE_RE = re.compile(r"^(?:F|NC|VULN)-\d{1,4}\b", re.I)

SEVERITY_TERMS = {
    "critical": 1.0,
    "high": 0.8,
    "medium": 0.5,
    "moderate": 0.5,
    "low": 0.25,
    "informational": 0.1,
    "info": 0.1,
}
SEV_RE = re.compile(r"\b(" + "|".join(SEVERITY_TERMS) + r")\b", re.I)
CVSS_RE = re.compile(r"CVSS(?:\s*v?\d(?:\.\d)?)?\s*[:\-]?\s*(\d{1,2}(?:\.\d)?)", re.I)


def detect_heading(line: str) -> Optional[str]:
    """Detect if a text line is a numbered or keyword section heading.

    Args:
        line: Text line to inspect.

    Returns:
        Heading title if detected, else None.
    """
    line = line.strip()
    if not line or len(line) > 90 or "|" in line:
        return None

    m = NUMBERED_HEADING.match(line)
    if m and len(line.split()) <= 10:
        return m.group(2).strip()

    if (
        KEYWORD_HEADING.match(line)
        and len(line.split()) <= 6
        and not line.endswith((".", ",", ";"))
    ):
        return line.strip()

    return None


def classify_section(title: str) -> str:
    """Classify a heading title into one of the canonical section types.

    Args:
        title: Heading text string.

    Returns:
        One of the 8 canonical section types (exec_summary, scope, etc.).
    """
    t = title.lower()
    for stype, keys in SECTION_KEYWORDS:
        if any(k in t for k in keys):
            return stype
    return "other"


def severity_score(text: str) -> float:
    """Extract maximum severity score from severity terms and CVSS scores in text.

    Args:
        text: Text string to inspect.

    Returns:
        Float score in [0.0, 1.0].
    """
    scores = [SEVERITY_TERMS[m.group(1).lower()] for m in SEV_RE.finditer(text)]
    for m in CVSS_RE.finditer(text):
        try:
            v = float(m.group(1))
            if 0.0 <= v <= 10.0:
                scores.append(round(v / 10.0, 4))
        except ValueError:
            pass
    return max(scores) if scores else 0.0


def severity_to_level(score: float) -> str:
    """Map numeric severity score in [0.0, 1.0] to categorical level string.

    Args:
        score: Severity score.

    Returns:
        Categorical string ('critical', 'high', 'medium', 'low', or 'none').
    """
    if score >= 0.9:
        return "critical"
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "medium"
    if score > 0.0:
        return "low"
    return "none"


def extract_finding_id(text: str) -> Optional[str]:
    """Extract and normalize a finding identifier (e.g. F-1, Finding 3 -> F-3).

    Args:
        text: Text string.

    Returns:
        Canonical finding ID (e.g. 'F-3') or None.
    """
    m = FID_WORD_RE.search(text) or FID_CODE_RE.search(text)
    if not m:
        return None
    return f"F-{int(m.group(1))}"


def is_table_line(line: str) -> bool:
    """Check if a line represents a table row.

    Args:
        line: Text line.

    Returns:
        True if the line appears to be part of a table.
    """
    if line.count("|") >= 2:
        return True
    if len(re.findall(r"\S {3,}\S", line)) >= 2:
        return True
    return bool(FINDING_LINE_RE.match(line))
