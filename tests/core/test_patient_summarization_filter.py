# Author: Leo
# NetID: lab8
# Description: Fast tests for the unsupported-fact filter using only made-up
#     (synthetic) text. No real MIMIC data is used.

import unittest

from pyhealth.tasks.patient_summarization_filter import (
    UnsupportedFactFilter,
    normalize_number,
)

MEDICATIONS = ("aspirin", "metoprolol", "lisinopril")

CONTEXT = (
    "Mr. ___ was admitted with chest pain. Troponin was negative. He was "
    "started on aspirin 81 mg daily and metoprolol 25 mg twice daily. "
    "Stress test was normal."
)

SUMMARY = (
    "Dear Mr. ___, You were admitted for chest pain. Your heart tests were "
    "normal. Please take aspirin 81 mg daily and metoprolol 25 mg twice daily. "
    "Take lisinopril 10 mg daily. Please follow up with your cardiologist in "
    "2 weeks."
)


class TestFind(unittest.TestCase):
    def test_flags_unsupported_medication_and_numbers(self):
        fact_filter = UnsupportedFactFilter(medication_names=MEDICATIONS)
        spans = fact_filter.find(SUMMARY, CONTEXT)
        self.assertEqual(
            [(s["text"], s["label"]) for s in spans],
            [("lisinopril", "medication"), ("10", "number"), ("2", "number")],
        )

    def test_offsets_point_into_the_summary(self):
        spans = UnsupportedFactFilter(medication_names=MEDICATIONS).find(
            SUMMARY, CONTEXT
        )
        self.assertEqual(
            [(s["start"], s["end"]) for s in spans],
            [(149, 159), (160, 162), (216, 217)],
        )
        for span in spans:
            self.assertEqual(SUMMARY[span["start"] : span["end"]], span["text"])

    def test_fully_supported_summary_gives_no_spans(self):
        fact_filter = UnsupportedFactFilter(medication_names=MEDICATIONS)
        self.assertEqual(fact_filter.find("Take aspirin 81 mg.", CONTEXT), [])

    def test_empty_summary_gives_no_spans(self):
        fact_filter = UnsupportedFactFilter(medication_names=MEDICATIONS)
        self.assertEqual(fact_filter.find("", CONTEXT), [])


class TestNumbers(unittest.TestCase):
    def test_normalize_number(self):
        cases = {"1,000": "1000", "5.0": "5", "2.50": "2.5", "007": "7", "0.5": "0.5"}
        for written, expected in cases.items():
            with self.subTest(written=written):
                self.assertEqual(normalize_number(written), expected)

    def test_equal_values_written_differently_are_supported(self):
        spans = UnsupportedFactFilter().find(
            "Take 5.0 mg and 1,000 units.", "Gave 5 mg and 1000 units."
        )
        self.assertEqual(spans, [])

    def test_different_value_is_flagged(self):
        spans = UnsupportedFactFilter().find("Take 2.5 mg.", "Gave 25 mg.")
        self.assertEqual([s["text"] for s in spans], ["2.5"])


class TestMedications(unittest.TestCase):
    def test_match_ignores_case(self):
        fact_filter = UnsupportedFactFilter(medication_names=["aspirin"])
        self.assertEqual(fact_filter.find("Take Aspirin.", "aspirin given"), [])
        spans = fact_filter.find("Take ASPIRIN.", "no drugs given")
        self.assertEqual([s["text"] for s in spans], ["ASPIRIN"])

    def test_name_inside_a_longer_word_is_not_a_match(self):
        fact_filter = UnsupportedFactFilter(medication_names=["insulin"])
        self.assertEqual(fact_filter.find("No sign of insulinoma.", ""), [])

    def test_no_medication_names_checks_numbers_only(self):
        spans = UnsupportedFactFilter().find(SUMMARY, CONTEXT)
        self.assertEqual([s["text"] for s in spans], ["10", "2"])


class TestKinds(unittest.TestCase):
    def test_numbers_only(self):
        fact_filter = UnsupportedFactFilter(
            kinds=("number",), medication_names=MEDICATIONS
        )
        spans = fact_filter.find(SUMMARY, CONTEXT)
        self.assertEqual([s["label"] for s in spans], ["number", "number"])

    def test_medications_only(self):
        fact_filter = UnsupportedFactFilter(
            kinds=("medication",), medication_names=MEDICATIONS
        )
        spans = fact_filter.find(SUMMARY, CONTEXT)
        self.assertEqual([s["text"] for s in spans], ["lisinopril"])

    def test_unknown_kind_raises(self):
        with self.assertRaises(ValueError):
            UnsupportedFactFilter(kinds=("number", "dose"))


class TestOverlaps(unittest.TestCase):
    def test_number_inside_a_medication_name_is_dropped(self):
        fact_filter = UnsupportedFactFilter(medication_names=["vitamin B12"])
        spans = fact_filter.find("Take vitamin B12 daily.", "")
        self.assertEqual(
            [(s["text"], s["label"]) for s in spans],
            [("vitamin B12", "medication")],
        )


class TestDropSentences(unittest.TestCase):
    def setUp(self):
        self.fact_filter = UnsupportedFactFilter(medication_names=MEDICATIONS)

    def test_drops_sentences_that_hold_a_span(self):
        spans = self.fact_filter.find(SUMMARY, CONTEXT)
        self.assertEqual(
            self.fact_filter.drop_sentences(SUMMARY, spans),
            "Dear Mr. ___, You were admitted for chest pain. Your heart tests were "
            "normal. Please take aspirin 81 mg daily and metoprolol 25 mg twice "
            "daily.",
        )

    def test_no_spans_keeps_the_summary(self):
        self.assertEqual(self.fact_filter.drop_sentences(SUMMARY, []), SUMMARY)

    def test_decimal_point_does_not_end_a_sentence(self):
        summary = "Take 2.5 mg daily. Rest at home."
        spans = self.fact_filter.find(summary, "")
        self.assertEqual(
            self.fact_filter.drop_sentences(summary, spans), "Rest at home."
        )

    def test_all_sentences_flagged_gives_empty_text(self):
        summary = "Take 10 mg. Come back in 2 days."
        spans = self.fact_filter.find(summary, "")
        self.assertEqual(self.fact_filter.drop_sentences(summary, spans), "")

    def test_empty_summary_gives_empty_text(self):
        self.assertEqual(self.fact_filter.drop_sentences("", []), "")


if __name__ == "__main__":
    unittest.main()
