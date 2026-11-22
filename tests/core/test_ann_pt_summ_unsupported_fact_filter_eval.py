# Author: Leo
# NetID: lab8
# Description: Fast tests for the ann-pt-summ evaluation script, using a small
#     made-up file in the same JSONL shape as the expert-labeled data. No real
#     MIMIC or ann-pt-summ data is used.

import json
import tempfile
import unittest
from pathlib import Path

from examples.ann_pt_summ_unsupported_fact_filter_eval import evaluate, load_examples
from pyhealth.tasks.patient_summarization_filter import UnsupportedFactFilter


def make_label(summary: str, text: str, label: str) -> dict:
    """Builds an expert label for the first ``text`` in ``summary``."""
    start = summary.index(text)
    return {"start": start, "end": start + len(text), "label": label, "text": text}


SUMMARY_A = "Take aspirin 81 mg and lisinopril 10 mg. See cardiology in 2 weeks."
SUMMARY_B = "Rest at home. Drink water."

EXAMPLES = [
    {
        "text": "Started on aspirin 81 mg.",
        "summary": SUMMARY_A,
        "labels": [
            make_label(SUMMARY_A, "lisinopril 10 mg", "medication_unsupported"),
            make_label(SUMMARY_A, "cardiology", "name_unsupported"),
            make_label(SUMMARY_A, "2 weeks", "time_unsupported"),
        ],
    },
    {"text": "Patient rested.", "summary": SUMMARY_B, "labels": []},
]


class TestLoadExamples(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "labels.jsonl"
        self.path.write_text("".join(json.dumps(ex) + "\n" for ex in EXAMPLES))

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_one_example_per_line_in_order(self):
        self.assertEqual(load_examples(self.path), EXAMPLES)


class TestEvaluate(unittest.TestCase):
    def test_worked_example(self):
        # Flags in A: lisinopril, 10, 2 (81 is in the context). lisinopril is
        # partial in "lisinopril 10 mg", 10 is spurious because that expert
        # span is used, 2 is partial in "2 weeks", cardiology is missed.
        # B has no flags and no labels.
        fact_filter = UnsupportedFactFilter(medication_names=["aspirin", "lisinopril"])
        scores = evaluate(EXAMPLES, fact_filter)
        self.assertEqual(
            [scores[k] for k in ("correct", "partial", "spurious", "missed")],
            [0, 2, 1, 1],
        )
        for k in ("precision", "recall", "f1"):
            with self.subTest(score=k):
                self.assertAlmostEqual(scores[k], 1 / 3)


if __name__ == "__main__":
    unittest.main()
