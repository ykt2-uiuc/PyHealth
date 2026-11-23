# Author: Charles Tung Fang
# NetID: ykt2
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Patient summarization task on MIMIC-IV-Note discharge notes.
#     The Brief Hospital Course is the input and the Discharge Instructions
#     section is the patient-facing target summary.

import re
from typing import Any, ClassVar

from pyhealth.data.data import Patient
from pyhealth.processors import TextProcessor

from .base_task import BaseTask

# Top-level section headers of a MIMIC-IV discharge note. A whitelist is used
# so that in-text labels such as "ACTIVE ISSUES:" inside the hospital course
# are not mistaken for the start of a new section.
SECTION_HEADERS = (
    "Chief Complaint",
    "Major Surgical or Invasive Procedure",
    "History of Present Illness",
    "Past Medical History",
    "Social History",
    "Family History",
    "Physical Exam",
    "Pertinent Results",
    "Brief Hospital Course",
    "Medications on Admission",
    "Discharge Medications",
    "Discharge Disposition",
    "Discharge Diagnosis",
    "Discharge Condition",
    "Discharge Instructions",
    "Followup Instructions",
)

_HEADER_RE = re.compile(
    r"^(" + "|".join(re.escape(h) for h in SECTION_HEADERS) + r"):[ \t]*$",
    flags=re.MULTILINE,
)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_DEID_TOKEN = "___"

CONTEXT_HEADER = "Brief Hospital Course"
SUMMARY_HEADER = "Discharge Instructions"


def split_sections(text: str) -> dict[str, str]:
    """Splits a discharge note into its top-level sections.

    Args:
        text: The full text of one discharge note.

    Returns:
        A dict mapping each section header found in ``text`` to its body.
        If a header appears more than once, the first occurrence is kept.

    Examples:
        >>> note = "Brief Hospital Course:\\nOk.\\nDischarge Instructions:\\nRest."
        >>> split_sections(note)["Discharge Instructions"]
        'Rest.'
    """
    matches = list(_HEADER_RE.finditer(text))
    sections: dict[str, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.setdefault(match.group(1), text[match.end() : end].strip())
    return sections


def get_section(text: str, header: str) -> str | None:
    """Returns the body of one section, or ``None`` if it is missing or empty.

    Args:
        text: The full text of one discharge note.
        header: One of ``SECTION_HEADERS``.

    Returns:
        The stripped section body, or ``None``.

    Examples:
        >>> note = "Brief Hospital Course:\\nStable.\\nDischarge Instructions:\\nRest."
        >>> get_section(note, "Brief Hospital Course")
        'Stable.'
        >>> get_section(note, "Physical Exam") is None
        True
    """
    return split_sections(text).get(header) or None


def count_sentences(text: str) -> int:
    """Counts sentences with a simple punctuation-based splitter.

    Args:
        text: Any text.

    Returns:
        The number of non-empty sentences.

    Examples:
        >>> count_sentences("Rest at home. Drink water! Call us if worse?")
        3
    """
    return len([s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s])


def deid_per_word(text: str) -> float:
    """Returns the number of de-identification placeholders per word.

    Args:
        text: Any text. MIMIC replaces protected information with ``___``.

    Returns:
        ``count("___") / number_of_words``, or 0.0 for empty text.

    Examples:
        >>> deid_per_word("a ___ b ___")
        0.5
    """
    words = text.split()
    if not words:
        return 0.0
    return text.count(_DEID_TOKEN) / len(words)


class PatientSummarizationMIMIC4(BaseTask):
    """Patient summarization from MIMIC-IV-Note discharge notes.

    Works with :class:`~pyhealth.datasets.MIMIC4NoteDataset` (or
    :class:`~pyhealth.datasets.MIMIC4Dataset` with the ``discharge`` note
    table). Each discharge note yields one sample whose input is the Brief
    Hospital Course and whose target is the Discharge Instructions section,
    following the dataset construction of Hegselmann et al. (CHIL 2024).
    Notes missing either section, or failing a quality filter, are skipped.

    The defaults mirror the paper's quality filters: summaries shorter than
    350 characters or with fewer than 3 sentences are dropped, hospital
    courses shorter than 500 characters are dropped, and summaries with more
    than one de-identified field per 10 words are dropped.

    Args:
        min_context_chars: Minimum length of the hospital course.
        min_summary_chars: Minimum length of the discharge instructions.
        min_summary_sentences: Minimum number of sentences in the summary.
        max_context_chars: Optional maximum length of the hospital course
            (the paper uses 4000 for the subset annotated by experts).
        max_deid_per_word: Maximum ``___`` placeholders per word in the summary.
        **kwargs: Passed to :class:`~pyhealth.tasks.BaseTask`.

    Raises:
        ValueError: If a length or ratio argument is negative.

    Examples:
        >>> from pyhealth.datasets import MIMIC4NoteDataset
        >>> from pyhealth.tasks import PatientSummarizationMIMIC4
        >>> dataset = MIMIC4NoteDataset(
        ...     root="/path/to/mimic-iv-note/2.2", tables=["discharge"]
        ... )
        >>> task = PatientSummarizationMIMIC4(max_context_chars=4000)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "patient_summarization_mimic4"
    input_schema: ClassVar[dict[str, type[TextProcessor]]] = {"context": TextProcessor}
    output_schema: ClassVar[dict[str, type[TextProcessor]]] = {"summary": TextProcessor}

    def __init__(
        self,
        min_context_chars: int = 500,
        min_summary_chars: int = 350,
        min_summary_sentences: int = 3,
        max_context_chars: int | None = None,
        max_deid_per_word: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if min_context_chars < 0 or min_summary_chars < 0:
            raise ValueError("minimum lengths must be non-negative")
        if min_summary_sentences < 0:
            raise ValueError("min_summary_sentences must be non-negative")
        if max_context_chars is not None and max_context_chars < min_context_chars:
            raise ValueError("max_context_chars must be >= min_context_chars")
        if max_deid_per_word < 0:
            raise ValueError("max_deid_per_word must be non-negative")
        self.min_context_chars = min_context_chars
        self.min_summary_chars = min_summary_chars
        self.min_summary_sentences = min_summary_sentences
        self.max_context_chars = max_context_chars
        self.max_deid_per_word = max_deid_per_word

    def passes_filters(self, context: str, summary: str) -> bool:
        """Checks the quality filters for one (context, summary) pair.

        Args:
            context: The Brief Hospital Course text.
            summary: The Discharge Instructions text.

        Returns:
            True if the pair satisfies every configured filter.
        """
        if len(context) < self.min_context_chars:
            return False
        if self.max_context_chars is not None and len(context) > self.max_context_chars:
            return False
        if len(summary) < self.min_summary_chars:
            return False
        if count_sentences(summary) < self.min_summary_sentences:
            return False
        return deid_per_word(summary) <= self.max_deid_per_word

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Builds summarization samples from one patient's discharge notes.

        Args:
            patient: A patient with ``discharge`` events carrying a ``text``
                attribute.

        Returns:
            A list of samples with keys ``patient_id``, ``note_id``,
            ``context`` and ``summary``. Notes that are missing a section or
            fail a filter produce no sample.
        """
        samples = []
        for note in patient.get_events(event_type="discharge"):
            text = getattr(note, "text", None)
            if not isinstance(text, str):
                continue
            context = get_section(text, CONTEXT_HEADER)
            summary = get_section(text, SUMMARY_HEADER)
            if context is None or summary is None:
                continue
            if not self.passes_filters(context, summary):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "note_id": getattr(note, "note_id", None),
                    "context": context,
                    "summary": summary,
                }
            )
        return samples
