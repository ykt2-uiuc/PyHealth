# Author: Leo
# NetID: lab8
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Rule-based filter that flags facts in a patient summary that
#     the hospital course does not support: numbers and medication names.

import re
from collections.abc import Iterable, Sequence
from typing import Any

Span = dict[str, Any]

KINDS = ("number", "medication")

# A number such as 10, 2.5 or 1,000. Commas only count as thousands separators.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# The space after ".", "!" or "?" ends a sentence, except after a title such as "Mr.".
_SENTENCE_BREAK_RE = re.compile(
    r"(?<=[.!?])(?<!\bMr\.)(?<!\bMrs\.)(?<!\bMs\.)(?<!\bDr\.)\s+"
)


def normalize_number(text: str) -> str:
    """Returns one standard form of a number, so equal values compare equal.

    Args:
        text: A number as written, such as ``"1,000"`` or ``"5.0"``.

    Returns:
        The number without thousands separators, leading zeros or trailing
        decimal zeros.

    Examples:
        >>> normalize_number("1,000")
        '1000'
        >>> normalize_number("5.0")
        '5'
    """
    whole, _, fraction = text.replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return f"{whole}.{fraction}" if fraction else whole


class UnsupportedFactFilter:
    """Flags numbers and medication names in a summary that its context lacks.

    A fact is supported if the same number (after ``normalize_number``) or the
    same medication name (case-insensitive, whole word) appears anywhere in the
    context. Hegselmann et al. (CHIL 2024) show that unsupported facts in
    training summaries teach a model to hallucinate.

    Args:
        kinds: The facts to check: ``"number"``, ``"medication"`` or both.
        medication_names: Medication names to look for. If ``None``, no
            medications are checked.

    Raises:
        ValueError: If ``kinds`` holds an unknown kind.

    Examples:
        >>> fact_filter = UnsupportedFactFilter(medication_names=["lisinopril"])
        >>> spans = fact_filter.find("Take lisinopril 10 mg.", "No new drugs.")
        >>> [(s["text"], s["label"]) for s in spans]
        [('lisinopril', 'medication'), ('10', 'number')]
    """

    def __init__(
        self,
        kinds: Sequence[str] = KINDS,
        medication_names: Iterable[str] | None = None,
    ) -> None:
        unknown = set(kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown kinds: {sorted(unknown)}")
        self.kinds = tuple(kinds)
        self.medication_names = tuple(medication_names or ())
        self._medication_patterns = [
            re.compile(rf"\b{re.escape(name)}\b", flags=re.IGNORECASE)
            for name in self.medication_names
        ]

    def find(self, summary: str, context: str) -> list[Span]:
        """Finds the facts in ``summary`` that ``context`` does not support.

        Args:
            summary: The patient summary (Discharge Instructions).
            context: The source text (Brief Hospital Course).

        Returns:
            Spans ``{"start", "end", "text", "label"}`` sorted by ``start``,
            with no overlaps. ``start`` and ``end`` are character offsets into
            ``summary``; ``end`` is exclusive.

        Examples:
            >>> UnsupportedFactFilter().find("Rest for 3 days.", "Rest 3 days.")
            []
        """
        spans: list[Span] = []
        if "number" in self.kinds:
            spans += self._find_numbers(summary, context)
        if "medication" in self.kinds:
            spans += self._find_medications(summary, context)
        return _remove_overlaps(spans)

    def drop_sentences(self, summary: str, spans: list[Span]) -> str:
        """Removes every sentence of ``summary`` that holds a span.

        A sentence ends at ".", "!" or "?" followed by whitespace. A decimal
        point ("2.5") or a title ("Mr.", "Mrs.", "Ms.", "Dr.") does not end a
        sentence.

        Args:
            summary: The patient summary that ``spans`` point into.
            spans: Spans from :meth:`find` on the same ``summary``.

        Returns:
            The sentences without spans, joined by single spaces. An empty
            string if every sentence holds a span.

        Examples:
            >>> fact_filter = UnsupportedFactFilter()
            >>> summary = "Rest at home. Take 10 mg daily."
            >>> spans = fact_filter.find(summary, "Rest.")
            >>> fact_filter.drop_sentences(summary, spans)
            'Rest at home.'
        """
        kept = [
            summary[start:end]
            for start, end in _sentence_bounds(summary)
            if not any(s["start"] < end and s["end"] > start for s in spans)
        ]
        return " ".join(kept)

    def _find_numbers(self, summary: str, context: str) -> list[Span]:
        supported = {normalize_number(m.group()) for m in _NUMBER_RE.finditer(context)}
        return [
            _make_span(match, "number")
            for match in _NUMBER_RE.finditer(summary)
            if normalize_number(match.group()) not in supported
        ]

    def _find_medications(self, summary: str, context: str) -> list[Span]:
        spans = []
        for pattern in self._medication_patterns:
            if pattern.search(context):
                continue
            spans += [_make_span(m, "medication") for m in pattern.finditer(summary)]
        return spans


def _make_span(match: re.Match, label: str) -> Span:
    return {
        "start": match.start(),
        "end": match.end(),
        "text": match.group(),
        "label": label,
    }


def _sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Returns (start, end) offsets of each sentence, without outer whitespace."""
    start = len(text) - len(text.lstrip())
    stop = len(text.rstrip())
    bounds = []
    for match in _SENTENCE_BREAK_RE.finditer(text, start, stop):
        bounds.append((start, match.start()))
        start = match.end()
    if start < stop:
        bounds.append((start, stop))
    return bounds


def _remove_overlaps(spans: list[Span]) -> list[Span]:
    """Sorts spans by start and keeps the longer span when two overlap."""
    kept: list[Span] = []
    for span in sorted(spans, key=lambda s: (s["start"], s["start"] - s["end"])):
        if not kept or span["start"] >= kept[-1]["end"]:
            kept.append(span)
    return kept
