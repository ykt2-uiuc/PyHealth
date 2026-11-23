# Author: Charles Tung Fang
# NetID: ykt2
# Description: Fast tests for PatientSummarizationMIMIC4 using only made-up
#     (synthetic) discharge notes. No real MIMIC data is used.

import csv
import gzip
import os
import tempfile
import unittest
from datetime import datetime
from unittest.mock import patch

import polars as pl

from pyhealth.data import Patient
from pyhealth.datasets import MIMIC4NoteDataset
from pyhealth.tasks.patient_summarization_mimic4 import (
    PatientSummarizationMIMIC4,
    count_sentences,
    deid_per_word,
    get_section,
    split_sections,
)

GOOD_NOTE = """Name:  ___                Unit No:   ___

Chief Complaint:
Shortness of breath

Brief Hospital Course:
Mrs. ___ is a ___ female with a history of coronary artery disease, hypertension
and hyperlipidemia who presented to the ED with severe dyspnea and chest pain.
ACTIVE ISSUES:
# Acute on chronic diastolic heart failure: ECG showed no acute ST changes and
cardiac enzymes were negative twice. She was diuresed with IV furosemide with
good effect and her symptoms improved over several days.
# Pneumonia: developed a productive cough and a new oxygen requirement. A chest
x-ray showed a left lower lobe infiltrate. She was started on azithromycin.

Medications on Admission:
metoprolol 25 mg daily

Discharge Instructions:
You were admitted to the hospital because you were short of breath. You were
found to have fluid build-up in your lungs from heart failure. We gave you
medicine through your vein to remove this fluid and your breathing got better.
You also had a lung infection and we treated it with an antibiotic. Please take
your antibiotic every day until it is finished. Weigh yourself every morning and
call your doctor if you gain more than 3 pounds in one day.

Followup Instructions:
___
"""
SHORT_NOTE = "Brief Hospital Course:\nStable.\n\nDischarge Instructions:\nRest.\n"
NO_SECTIONS_NOTE = "This note has no recognizable headers at all."
NO_SUMMARY_NOTE = "Brief Hospital Course:\n" + "Stable course. " * 50 + "\n"


def make_note(context: str, summary: str) -> str:
    """Builds a minimal note from a hospital course and instructions."""
    return (
        f"Brief Hospital Course:\n{context}\n\n"
        f"Discharge Instructions:\n{summary}\n\nFollowup Instructions:\n___\n"
    )


def make_patient(patient_id: str, notes, event_type: str = "discharge") -> Patient:
    """Builds a real PyHealth Patient holding the given notes."""
    t = datetime(2020, 1, 1)
    df = pl.DataFrame(
        {
            "patient_id": [patient_id] * len(notes),
            "event_type": [event_type] * len(notes),
            "timestamp": [t] * len(notes),
            f"{event_type}/note_id": [f"n{i}" for i in range(len(notes))],
            f"{event_type}/text": notes,
        }
    )
    return Patient(patient_id=patient_id, data_source=df)


class TestSectionHelpers(unittest.TestCase):
    def test_finds_whitelisted_sections_in_order(self):
        sections = split_sections(GOOD_NOTE)
        self.assertEqual(
            list(sections),
            [
                "Chief Complaint",
                "Brief Hospital Course",
                "Medications on Admission",
                "Discharge Instructions",
                "Followup Instructions",
            ],
        )

    def test_inline_label_does_not_split_a_section(self):
        course = split_sections(GOOD_NOTE)["Brief Hospital Course"]
        self.assertIn("ACTIVE ISSUES:", course)
        self.assertIn("azithromycin", course)

    def test_section_bodies_are_exact(self):
        sections = split_sections(make_note("Course text.", "Rest at home."))
        self.assertEqual(sections["Brief Hospital Course"], "Course text.")
        self.assertEqual(sections["Discharge Instructions"], "Rest at home.")

    def test_missing_or_empty_section_returns_none(self):
        self.assertIsNone(get_section(NO_SECTIONS_NOTE, "Brief Hospital Course"))
        self.assertIsNone(
            get_section("Brief Hospital Course:\n\n", "Brief Hospital Course")
        )

    def test_duplicate_header_keeps_first(self):
        text = "Brief Hospital Course:\nfirst\nBrief Hospital Course:\nsecond\n"
        self.assertEqual(get_section(text, "Brief Hospital Course"), "first")

    def test_header_inside_a_sentence_is_ignored(self):
        text = "See the Brief Hospital Course: it was stable.\n"
        self.assertEqual(split_sections(text), {})

    def test_count_sentences(self):
        self.assertEqual(count_sentences(""), 0)
        self.assertEqual(count_sentences("One. Two! Three?"), 3)
        self.assertEqual(count_sentences("No end punctuation"), 1)

    def test_deid_per_word(self):
        self.assertEqual(deid_per_word(""), 0.0)
        self.assertAlmostEqual(deid_per_word("a ___ b ___"), 0.5)


class TestTaskCall(unittest.TestCase):
    def test_keeps_good_note_and_has_expected_fields(self):
        samples = PatientSummarizationMIMIC4()(make_patient("p1", [GOOD_NOTE]))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(set(sample), {"patient_id", "note_id", "context", "summary"})
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["note_id"], "n0")
        self.assertTrue(sample["context"].startswith("Mrs. ___ is a ___ female"))
        self.assertTrue(sample["summary"].startswith("You were admitted"))
        self.assertNotIn("Followup", sample["summary"])

    def test_skips_short_missing_and_incomplete_notes(self):
        notes = [GOOD_NOTE, SHORT_NOTE, NO_SECTIONS_NOTE, NO_SUMMARY_NOTE]
        samples = PatientSummarizationMIMIC4()(make_patient("p1", notes))
        self.assertEqual([s["note_id"] for s in samples], ["n0"])

    def test_loose_filters_keep_short_note(self):
        task = PatientSummarizationMIMIC4(
            min_context_chars=0, min_summary_chars=0, min_summary_sentences=0
        )
        samples = task(make_patient("p1", [GOOD_NOTE, SHORT_NOTE]))
        self.assertEqual(len(samples), 2)

    def test_patient_without_discharge_events_gives_no_samples(self):
        patient = make_patient("p1", [GOOD_NOTE], event_type="radiology")
        self.assertEqual(PatientSummarizationMIMIC4()(patient), [])

    def test_non_string_text_is_skipped(self):
        patient = make_patient("p1", [None, GOOD_NOTE])
        samples = PatientSummarizationMIMIC4()(patient)
        self.assertEqual([s["note_id"] for s in samples], ["n1"])

    def test_multiple_notes_give_multiple_samples(self):
        samples = PatientSummarizationMIMIC4()(make_patient("p1", [GOOD_NOTE] * 3))
        self.assertEqual(len(samples), 3)


class TestFilters(unittest.TestCase):
    SUMMARY = "Rest at home. Drink water. Call us if you feel worse."

    def test_min_context_chars(self):
        task = PatientSummarizationMIMIC4(min_context_chars=10, min_summary_chars=0)
        self.assertFalse(task.passes_filters("too short", self.SUMMARY))
        self.assertTrue(task.passes_filters("long enough text", self.SUMMARY))

    def test_max_context_chars(self):
        task = PatientSummarizationMIMIC4(
            min_context_chars=0, min_summary_chars=0, max_context_chars=10
        )
        self.assertTrue(task.passes_filters("0123456789", self.SUMMARY))
        self.assertFalse(task.passes_filters("0123456789A", self.SUMMARY))

    def test_min_summary_chars(self):
        task = PatientSummarizationMIMIC4(min_context_chars=0, min_summary_chars=100)
        self.assertFalse(task.passes_filters("context", self.SUMMARY))

    def test_min_summary_sentences(self):
        task = PatientSummarizationMIMIC4(
            min_context_chars=0, min_summary_chars=0, min_summary_sentences=3
        )
        self.assertFalse(task.passes_filters("context", "One. Two."))
        self.assertTrue(task.passes_filters("context", "One. Two. Three."))

    def test_max_deid_per_word(self):
        task = PatientSummarizationMIMIC4(
            min_context_chars=0,
            min_summary_chars=0,
            min_summary_sentences=0,
            max_deid_per_word=0.1,
        )
        self.assertFalse(task.passes_filters("context", "a ___ b ___ c."))
        self.assertTrue(task.passes_filters("context", "a b c d e f g h i j ___."))

    def test_invalid_arguments_raise(self):
        bad_kwargs = [
            {"min_context_chars": -1},
            {"min_summary_chars": -1},
            {"min_summary_sentences": -1},
            {"max_deid_per_word": -0.1},
            {"min_context_chars": 100, "max_context_chars": 50},
        ]
        for kwargs in bad_kwargs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    PatientSummarizationMIMIC4(**kwargs)

    def test_schema_and_name(self):
        task = PatientSummarizationMIMIC4()
        self.assertEqual(task.task_name, "patient_summarization_mimic4")
        self.assertEqual(list(task.input_schema), ["context"])
        self.assertEqual(list(task.output_schema), ["summary"])


class TestWithMIMIC4NoteDataset(unittest.TestCase):
    """Runs the task through the real dataset on two made-up notes."""

    # Pretending to run inside a notebook makes PyHealth use threads instead of
    # starting worker processes, which keeps this test under one second.
    @patch("pyhealth.datasets.base_dataset.in_notebook", return_value=True)
    def test_set_task_on_synthetic_discharge_table(self, _mock_in_notebook):
        with (
            tempfile.TemporaryDirectory() as root,
            tempfile.TemporaryDirectory() as cache,
        ):
            os.makedirs(os.path.join(root, "note"))
            rows = [
                [
                    "n1",
                    10,
                    100,
                    "DS",
                    1,
                    "2150-01-01 10:00:00",
                    "2150-01-01 12:00:00",
                    GOOD_NOTE,
                ],
                [
                    "n2",
                    11,
                    101,
                    "DS",
                    1,
                    "2150-02-01 10:00:00",
                    "2150-02-01 12:00:00",
                    SHORT_NOTE,
                ],
            ]
            path = os.path.join(root, "note", "discharge.csv.gz")
            with gzip.open(path, "wt", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "note_id",
                        "subject_id",
                        "hadm_id",
                        "note_type",
                        "note_seq",
                        "charttime",
                        "storetime",
                        "text",
                    ]
                )
                writer.writerows(rows)
            dataset = MIMIC4NoteDataset(
                root=root, tables=["discharge"], cache_dir=cache
            )
            samples = dataset.set_task(PatientSummarizationMIMIC4())
            self.assertEqual(len(samples), 1)
            sample = samples[0]
            self.assertEqual(sample["patient_id"], "10")
            self.assertIn("coronary artery disease", sample["context"])
            self.assertIn("short of breath", sample["summary"])


if __name__ == "__main__":
    unittest.main()
