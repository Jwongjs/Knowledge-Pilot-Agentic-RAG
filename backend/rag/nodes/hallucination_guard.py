from __future__ import annotations
import re
from models.agent_state import Evidence

# Matches [1], [¹], [^1], **[1]**, or bare superscripts ¹²³ — all styles the LLM may produce
_CITATION_RE = re.compile(r"(?:\*{0,2}\[\^?([¹²³⁴⁵⁶⁷⁸⁹\d]+)\]\*{0,2}|([¹²³⁴⁵⁶⁷⁸⁹]+))")
# Factual signals: numbers/percentages only (not every capitalised word)
_FACTUAL_RE = re.compile(r"\b\d+[\.,]?\d*\s*%?\b")
_STOPWORDS = {"the", "a", "an", "is", "in", "of", "and", "or", "to", "for", "with"}
_ALIGNMENT_WINDOW = 100  # chars before/after a citation marker — robust to decimals like 18.9


def check_hallucination(answer: str, all_evidence: list[Evidence]) -> bool:
    """Return True if the answer is suspected of hallucinating (failed any check)."""
    valid_ids = {str(i + 1) for i in range(len(all_evidence))}
    return (
        not _check_citation_validity(answer, valid_ids)
        or not _check_citation_coverage(answer)
        or not _check_citation_alignment(answer, all_evidence)
    )


def _check_citation_validity(answer: str, valid_ids: set[str]) -> bool:
    cited = {g1 or g2 for g1, g2 in _CITATION_RE.findall(answer)}
    normalised = {c.translate(str.maketrans("¹²³⁴⁵⁶⁷⁸⁹", "123456789")) for c in cited if c}
    return normalised.issubset(valid_ids)


def _check_citation_coverage(answer: str) -> bool:
    sentences = re.split(r"(?<=[^\d\w])[.!?](?=\s|$)", answer)
    for sentence in sentences:
        if _FACTUAL_RE.search(sentence) and not _CITATION_RE.search(sentence):
            return False
    return True


def _check_citation_alignment(answer: str, evidence_list: list[Evidence]) -> bool:
    for match in _CITATION_RE.finditer(answer):
        raw = (match.group(1) or match.group(2) or "").translate(str.maketrans("¹²³⁴⁵⁶⁷⁸⁹", "123456789"))
        if not raw.isdigit():
            continue
        idx = int(raw) - 1
        if idx >= len(evidence_list):
            return False
        window = _surrounding_window(answer, match.start())
        chunk_tokens = set(evidence_list[idx].content.lower().split()) - _STOPWORDS
        window_tokens = set(window.lower().split()) - _STOPWORDS
        if not (chunk_tokens & window_tokens):
            return False
    return True


def _surrounding_window(text: str, pos: int, span: int = _ALIGNMENT_WINDOW) -> str:
    return text[max(0, pos - span): min(len(text), pos + span)]
