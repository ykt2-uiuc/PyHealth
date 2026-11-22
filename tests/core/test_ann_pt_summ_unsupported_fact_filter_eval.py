# Author: Leo
# NetID: lab8
# Description: Fast tests for the ann-pt-summ evaluation script, using a small
#     made-up file in the same JSONL shape as the expert-labeled data. No real
#     MIMIC or ann-pt-summ data is used.

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from examples.ann_pt_summ_unsupported_fact_filter_eval import (
    evaluate,
    load_examples,
    main,
)
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


class TestMain(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        tmp = Path(self.tmp.name)
        self.data = tmp / "labels.jsonl"
        self.data.write_text("".join(json.dumps(ex) + "\n" for ex in EXAMPLES))
        self.medications = tmp / "medications.txt"
        self.medications.write_text("aspirin\n\nlisinopril\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_main(self, *args: str) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            main(["--data", str(self.data), *args])
        return out.getvalue()

    def rows(self, output: str) -> list[list[str]]:
        """Splits each printed row after the header into its columns."""
        return [line.split() for line in output.splitlines()[1:]]

    def test_prints_one_row_per_filter_setting(self):
        # number: 10 and 2 are partial, cardiology is missed -> F1 40.0.
        # medication: lisinopril is partial, two spans missed -> F1 25.0.
        # both: the worked example -> F1 33.3.
        rows = self.rows(self.run_main("--medications", str(self.medications)))
        self.assertEqual([row[0] for row in rows], ["number", "medication", "both"])
        self.assertEqual([row[3] for row in rows], ["40.0", "25.0", "33.3"])

    def test_without_medications_checks_numbers_only(self):
        rows = self.rows(self.run_main())
        self.assertEqual([row[0] for row in rows], ["number"])

    def test_prints_no_text_from_the_data(self):
        output = self.run_main("--medications", str(self.medications))
        for word in ("lisinopril", "cardiology", "weeks", "aspirin"):
            with self.subTest(word=word):
                self.assertNotIn(word, output)


if __name__ == "__main__":
    unittest.main()
