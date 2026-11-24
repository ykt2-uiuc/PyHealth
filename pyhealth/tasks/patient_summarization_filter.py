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

# Common US generic names: ranks 1 to 130 of the ClinCalc DrugStats Top 300 of
# 2024 (https://clincalc.com/DrugStats, from the MEPS survey). Combinations are
# split into ingredients and all insulins are one name. Left out: names that plain
# text also uses for lab tests, nutrients or body parts, and extra salt forms.
# fmt: off
DEFAULT_MEDICATIONS = (
    "acetaminophen", "adalimumab", "albuterol", "alendronate", "allopurinol",
    "alprazolam", "amitriptyline", "amlodipine", "amoxicillin", "apixaban",
    "aripiprazole", "aspirin", "atenolol", "atorvastatin", "azelastine",
    "azithromycin", "baclofen", "budesonide", "bupropion", "buspirone",
    "carvedilol", "celecoxib", "cephalexin", "cetirizine", "chlorthalidone",
    "cholecalciferol", "citalopram", "clavulanate", "clonazepam", "clonidine",
    "clopidogrel", "cyanocobalamin", "cyclobenzaprine", "dapagliflozin",
    "dextroamphetamine", "diclofenac", "diltiazem", "doxycycline", "dulaglutide",
    "duloxetine", "empagliflozin", "ergocalciferol", "escitalopram", "esomeprazole",
    "ezetimibe", "famotidine", "fenofibrate", "finasteride", "fluoxetine",
    "fluticasone", "formoterol", "furosemide", "gabapentin", "glimepiride",
    "glipizide", "hydralazine", "hydrochlorothiazide", "hydrocodone", "hydroxyzine",
    "ibuprofen", "insulin", "lamotrigine", "latanoprost", "levetiracetam",
    "levothyroxine", "lisdexamfetamine", "lisinopril", "loratadine", "lorazepam",
    "losartan", "meloxicam", "metformin", "methocarbamol", "methylphenidate",
    "metoprolol", "mirtazapine", "montelukast", "naproxen", "nifedipine",
    "norethindrone", "norgestimate", "olmesartan", "omeprazole", "ondansetron",
    "oxybutynin", "oxycodone", "pantoprazole", "paroxetine", "pravastatin",
    "prednisolone", "prednisone", "pregabalin", "propranolol", "quetiapine",
    "rivaroxaban", "rosuvastatin", "salmeterol", "semaglutide", "sertraline",
    "simvastatin", "sitagliptin", "spironolactone", "sumatriptan", "tadalafil",
    "tamsulosin", "tirzepatide", "tizanidine", "topiramate", "tramadol",
    "trazodone", "triamcinolone", "umeclidinium", "valacyclovir", "valsartan",
    "venlafaxine", "vilanterol", "warfarin", "zolpidem",
)
# fmt: on

# A number such as 10, 2.5 or 1,000. Commas only count as thousands separators.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")

# Words and runs of punctuation, split as NLTK's ``wordpunct_tokenize`` does.
_WORD_RE = re.compile(r"\w+|[^\w\s]+")

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

    ``DEFAULT_MEDICATIONS`` holds common generic names only. It does not match a
    brand name to its generic name, and it lacks most drugs that are given only
    in a hospital.

    Args:
        kinds: The facts to check: ``"number"``, ``"medication"`` or both.
        medication_names: Medication names to look for, in place of
            ``DEFAULT_MEDICATIONS``. If ``None``, the default names are used.

    Raises:
        ValueError: If ``kinds`` holds an unknown kind.

    Examples:
        >>> fact_filter = UnsupportedFactFilter(medication_names=["lisinopril"])
        >>> spans = fact_filter.find("Take lisinopril 10 mg.", "No new drugs.")
        >>> [(s["text"], s["label"]) for s in spans]
        [('lisinopril', 'medication'), ('10', 'number')]

        Add names to the default list:

        >>> names = [*DEFAULT_MEDICATIONS, "vancomycin"]
        >>> fact_filter = UnsupportedFactFilter(medication_names=names)
        >>> spans = fact_filter.find("Take vancomycin and metformin.", "")
        >>> [s["text"] for s in spans]
        ['vancomycin', 'metformin']
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
        if medication_names is None:
            medication_names = DEFAULT_MEDICATIONS
        self.medication_names = tuple(medication_names)
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


def span_scores(
    summaries: Sequence[str],
    pred: Sequence[list[Span]],
    gold: Sequence[list[Span]],
) -> dict[str, float]:
    """Scores predicted spans against expert spans with partial matching.

    Follows the evaluation code of Hegselmann et al. (CHIL 2024): the
    "partial" scheme of ``nervaluate`` 0.1.8 (SemEval-2013 Task 9.1) on word
    positions, with labels ignored. Character offsets become word positions
    with the split of NLTK's ``wordpunct_tokenize``. For each predicted span,
    in order: the same words as an expert span count as correct; else the
    first overlapping expert span that no other prediction used counts as
    partial; else the prediction is spurious. Expert spans with no match are
    missed. As in ``nervaluate``, a span's last word position is inclusive, so
    spans on neighbouring words overlap. Counts are summed over all summaries.

    Args:
        summaries: The summaries the spans point into.
        pred: Predicted spans, one list per summary.
        gold: Expert spans, one list per summary.

    Returns:
        ``precision`` and ``recall`` (correct plus half of partial, divided by
        the predicted or the expert spans), ``f1``, and the counts
        ``correct``, ``partial``, ``spurious`` and ``missed``.

    Raises:
        ValueError: If the three sequences have different lengths.

    Examples:
        >>> summary = "Take lisinopril 10 mg."
        >>> pred = [[{"start": 5, "end": 15, "label": "medication"}]]
        >>> gold = [[{"start": 5, "end": 21, "label": "medication_unsupported"}]]
        >>> span_scores([summary], pred, gold)["f1"]
        0.5
    """
    if not len(summaries) == len(pred) == len(gold):
        raise ValueError("summaries, pred and gold must have the same length")
    counts = dict.fromkeys(("correct", "partial", "spurious", "missed"), 0)
    for summary, pred_spans, gold_spans in zip(summaries, pred, gold):
        gold_words = [_word_positions(summary, span) for span in gold_spans]
        used: set[int] = set()
        for p_start, p_end in (_word_positions(summary, s) for s in pred_spans):
            outcome = "spurious"
            for i, (g_start, g_end) in enumerate(gold_words):
                if (p_start, p_end) == (g_start, g_end):
                    outcome = "correct"
                elif i not in used and p_start <= g_end and g_start <= p_end:
                    outcome = "partial"
                else:
                    continue
                used.add(i)
                break
            counts[outcome] += 1
        counts["missed"] += len(gold_words) - len(used)

    matched = counts["correct"] + 0.5 * counts["partial"]
    predicted = counts["correct"] + counts["partial"] + counts["spurious"]
    expected = counts["correct"] + counts["partial"] + counts["missed"]
    precision = matched / predicted if predicted else 0.0
    recall = matched / expected if expected else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, **counts}


def _word_positions(summary: str, span: Span) -> tuple[int, int]:
    """Converts a character span to (start, end) word positions, end exclusive."""
    start = len(_WORD_RE.findall(summary[: span["start"]]))
    length = len(_WORD_RE.findall(summary[span["start"] : span["end"]]))
    return start, start + length


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
