"""
Gemma-based extractive paragraph summarization of security audit reports.

Produces a structured 2-to-3 paragraph executive summary:
  - Paragraph 1: Overview & Audit Context (scope, methodology, overall posture)
  - Paragraph 2: Key Findings & Impacts (critical, high, and medium severity findings)
  - Paragraph 3: Recommendations & Mitigation Actions (remediation steps)

Every single sentence in every paragraph is verified to be 100% VERBATIM from
the source document text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from summarizer.json_report import ExtractiveSentence, ReportDocument
from summarizer.llm_client import GemmaClient, GemmaError
from summarizer.selection import verify_verbatim_summary

logger = logging.getLogger("llm_summary")

MAX_PROMPT_CHARS = 300_000

PARAGRAPH_EXTRACTIVE_INSTRUCTIONS = """You are an expert cybersecurity auditor producing an executive summary for an evaluation body.

Below are candidate verbatim sentences extracted from an audit report, organized into three sections:
- Section A: Overview & Security Posture (background, scope, overall assessment)
- Section B: Key Findings & Impacts (observations and technical risks, with severity ratings)
- Section C: Recommendations & Remediation (corrective actions)

Select the most informative verbatim sentences to compose a cohesive, well-flowing 2-to-3 paragraph executive summary:
- PARAGRAPH 1 (Overview & Security Posture): Choose 2-4 sentences from Section A covering audit scope, methodology standards, and overall mitigation status.
- PARAGRAPH 2 (Key Findings & Impacts): Choose 2-5 sentences from Section B covering the highest-severity vulnerabilities (Critical, High, Medium) and their security impacts.
- PARAGRAPH 3 (Recommendations & Actions): Choose 1-3 sentences from Section C detailing key corrective actions.

Requirements:
1. Select ONLY sentence numbers that appear in the candidate list.
2. Ensure the selected sentences read smoothly as cohesive paragraphs.
3. Prioritize high/medium severity findings.
4. Do not select redundant sentences.

Respond in EXACTLY this format with no other text:
PARAGRAPH 1:
[list of sentence numbers, e.g. 1, 2, 3]

PARAGRAPH 2:
[list of sentence numbers, e.g. 5, 8, 12]

PARAGRAPH 3:
[list of sentence numbers, e.g. 15, 18]"""


def _clean_ids(raw_str: str) -> List[int]:
    """Parse comma or space separated numbers from a bracketed string like '[1, 2, 3]'."""
    nums: List[int] = []
    for token in re.findall(r"\b\d+\b", raw_str):
        try:
            nums.append(int(token))
        except ValueError:
            pass
    return nums


def parse_paragraph_selections(text: str) -> Dict[int, List[int]]:
    """Parse PARAGRAPH 1, 2, 3 sentence ID lists from model reply."""
    selections: Dict[int, List[int]] = {1: [], 2: [], 3: []}

    # Match PARAGRAPH X: [...]
    p_pattern = re.compile(
        r"PARAGRAPH\s*([123])\s*:\s*(?:\[(.*?)\]|([0-9,\s]+))",
        re.I,
    )
    for m in p_pattern.finditer(text):
        p_num = int(m.group(1))
        content = m.group(2) if m.group(2) is not None else m.group(3)
        ids = _clean_ids(content)
        if ids:
            selections[p_num] = ids

    # If format was JSON like {"paragraph_1": [1, 2], ...}
    if not any(selections.values()):
        try:
            m_json = re.search(r"\{.*\}", text, re.S)
            if m_json:
                data = json.loads(m_json.group(0))
                for key, val in data.items():
                    k_lower = key.lower()
                    for p_num in (1, 2, 3):
                        if str(p_num) in k_lower and isinstance(val, list):
                            selections[p_num] = [int(x) for x in val if isinstance(x, (int, str)) and str(x).isdigit()]
        except Exception:
            pass

    return selections


def render_candidate_listing(doc: ReportDocument, max_findings: int = 50, max_recs: int = 30) -> str:
    """Format candidate sentences grouped by section for the model prompt."""
    lines: List[str] = []

    # Section A: Overview
    lines.append("[Section A: Overview & Security Posture]")
    if doc.overview_sentences:
        for s in doc.overview_sentences:
            lines.append(f"[{s.uid}] {s.text}")
    else:
        lines.append("(No explicit narrative section in report)")

    # Section B: Findings & Impacts
    lines.append("\n[Section B: Key Findings & Impacts]")
    sorted_findings = sorted(
        doc.finding_sentences,
        key=lambda s: (s.severity_score, len(s.text)),
        reverse=True,
    )[:max_findings]
    if sorted_findings:
        for s in sorted_findings:
            sev_part = f"Severity: {s.severity}"
            title_part = f"Finding: {s.finding_title}" if s.finding_title else ""
            status_part = f"Status: {s.status}" if s.status else ""
            meta_str = " | ".join(p for p in [sev_part, title_part, status_part] if p)
            lines.append(f"[{s.uid}] ({meta_str}) {s.text}")
    else:
        lines.append("(No finding observations or impacts in report)")

    # Section C: Recommendations
    lines.append("\n[Section C: Recommendations & Remediation]")
    sorted_recs = sorted(
        doc.recommendation_sentences,
        key=lambda s: (s.severity_score, len(s.text)),
        reverse=True,
    )[:max_recs]
    if sorted_recs:
        for s in sorted_recs:
            sev_part = f"Severity: {s.severity}" if s.severity != "none" else ""
            title_part = f"Finding: {s.finding_title}" if s.finding_title else ""
            meta_str = " | ".join(p for p in [title_part, sev_part] if p)
            meta_tag = f" ({meta_str})" if meta_str else ""
            lines.append(f"[{s.uid}]{meta_tag} {s.text}")
    else:
        lines.append("(No explicit recommendation section in report)")

    return "\n".join(lines)


def llm_extractive_paragraphs(
    client: GemmaClient,
    doc: ReportDocument,
    target_words: int = 250,
) -> Tuple[str, Dict[str, Any]]:
    """Produce a 2-3 paragraph extractive summary using Gemma with deterministic verification.

    Args:
        client: Local Gemma client.
        doc: ReportDocument containing candidate sentences.
        target_words: Soft target word count for the complete summary.

    Returns:
        (summary_text, metadata_dict)
    """
    id_map: Dict[int, ExtractiveSentence] = {s.uid: s for s in doc.all_sentences}
    overview_uids = {s.uid for s in doc.overview_sentences}
    finding_uids = {s.uid for s in doc.finding_sentences}
    rec_uids = {s.uid for s in doc.recommendation_sentences}

    candidates_text = render_candidate_listing(doc)
    prompt = (
        f"{PARAGRAPH_EXTRACTIVE_INSTRUCTIONS}\n\n"
        f"REPORT: {doc.title}\n"
        f"CANDIDATE SENTENCES:\n{candidates_text}"
    )

    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS]

    # Generate model response
    raw_reply = client.chat(prompt, max_new_tokens=512)
    selections = parse_paragraph_selections(raw_reply)

    p1_sents: List[ExtractiveSentence] = []
    p2_sents: List[ExtractiveSentence] = []
    p3_sents: List[ExtractiveSentence] = []

    # Map model selections to validated sentences with deduplication and budget controls
    seen_texts: set[str] = set()

    def _is_dup(t: str) -> bool:
        t_clean = re.sub(r"[^a-z0-9 ]+", "", t.lower()).strip()
        if t_clean in seen_texts:
            return True
        for prev in seen_texts:
            if len(t_clean) > 30 and (t_clean in prev or prev in t_clean):
                return True
        seen_texts.add(t_clean)
        return False

    # Paragraph 1: Overview budget (~85 words max)
    w1 = 0
    for uid in selections.get(1, []):
        if uid in id_map:
            s = id_map[uid]
            if not _is_dup(s.text):
                if w1 + s.words <= 95 or not p1_sents:
                    p1_sents.append(s)
                    w1 += s.words
                if w1 >= 85:
                    break

    # Paragraph 2: Key Findings budget (~140 words max, prioritizing distinct findings)
    w2 = 0
    seen_fids = set()
    for uid in selections.get(2, []):
        if uid in id_map:
            s = id_map[uid]
            if not _is_dup(s.text):
                fid = s.finding_id or s.finding_title
                if fid not in seen_fids or len(seen_fids) < 3:
                    if w2 + s.words <= 150 or not p2_sents:
                        p2_sents.append(s)
                        w2 += s.words
                        if fid:
                            seen_fids.add(fid)
                if w2 >= 135:
                    break

    # Paragraph 3: Recommendations budget (~85 words max)
    w3 = 0
    for uid in selections.get(3, []):
        if uid in id_map:
            s = id_map[uid]
            if not _is_dup(s.text):
                if w3 + s.words <= 95 or not p3_sents:
                    p3_sents.append(s)
                    w3 += s.words
                if w3 >= 80:
                    break

    # Fallback enrichment if any section was empty
    # Paragraph 1 fallback
    if not p1_sents and doc.overview_sentences:
        w = 0
        for s in doc.overview_sentences:
            if w + s.words <= 90 or not p1_sents:
                p1_sents.append(s)
                w += s.words
            if w >= 75:
                break

    # Paragraph 2 fallback (Key Findings)
    if not p2_sents and doc.finding_sentences:
        sorted_findings = sorted(
            doc.finding_sentences,
            key=lambda s: (s.severity_score, len(s.text)),
            reverse=True,
        )
        w = 0
        seen_fids = set()
        for s in sorted_findings:
            fid = s.finding_id or s.finding_title
            if fid not in seen_fids or len(seen_fids) < 3:
                if w + s.words <= 140 or not p2_sents:
                    p2_sents.append(s)
                    w += s.words
                    if fid:
                        seen_fids.add(fid)
            if w >= 120:
                break

    # Paragraph 3 fallback (Recommendations)
    if not p3_sents and doc.recommendation_sentences:
        sorted_recs = sorted(
            doc.recommendation_sentences,
            key=lambda s: (s.severity_score, len(s.text)),
            reverse=True,
        )
        w = 0
        seen_rfids = set()
        for s in sorted_recs:
            fid = s.finding_id or s.finding_title
            if fid not in seen_rfids or len(seen_rfids) < 3:
                if w + s.words <= 90 or not p3_sents:
                    p3_sents.append(s)
                    w += s.words
                    if fid:
                        seen_rfids.add(fid)
            if w >= 75:
                break

    # Sort sentences within each paragraph by document order (uid) to preserve natural flow
    p1_sents.sort(key=lambda s: s.uid)
    p2_sents.sort(key=lambda s: s.uid)
    p3_sents.sort(key=lambda s: s.uid)

    def _term(t: str) -> str:
        t = t.strip()
        return t if (not t or t[-1] in ".!?;:") else t + "."

    # Combine into cohesive paragraphs
    paragraphs: List[str] = []
    if p1_sents:
        paragraphs.append(" ".join(_term(s.text) for s in p1_sents))
    if p2_sents:
        paragraphs.append(" ".join(_term(s.text) for s in p2_sents))
    if p3_sents:
        paragraphs.append(" ".join(_term(s.text) for s in p3_sents))

    summary_text = "\n\n".join(paragraphs)

    all_selected = p1_sents + p2_sents + p3_sents

    # Strict assertion: every sentence MUST be a verbatim substring of source text
    if doc.source_text:
        verify_verbatim_summary(all_selected, doc.source_text)

    info = {
        "n_paragraphs": len(paragraphs),
        "total_words": len(summary_text.split()),
        "p1_words": sum(s.words for s in p1_sents),
        "p2_words": sum(s.words for s in p2_sents),
        "p3_words": sum(s.words for s in p3_sents),
        "selected_uids": [s.uid for s in all_selected],
        "raw_response": raw_reply,
    }

    return summary_text, all_selected, info