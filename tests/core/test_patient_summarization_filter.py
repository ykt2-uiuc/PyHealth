# Author: Leo
# NetID: lab8
# Description: Fast tests for the unsupported-fact filter using only made-up
#     (synthetic) text. No real MIMIC data is used.

import unittest

from pyhealth.tasks.patient_summarization_filter import (
    DEFAULT_MEDICATIONS,
    UnsupportedFactFilter,
    normalize_number,
    span_scores,
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

    def test_default_list_flags_a_common_medication(self):
        spans = UnsupportedFactFilter().find("Take metformin.", "Gave aspirin.")
        self.assertEqual(
            [(s["text"], s["label"]) for s in spans], [("metformin", "medication")]
        )

    def test_default_list_is_clean(self):
        self.assertEqual(len(set(DEFAULT_MEDICATIONS)), len(DEFAULT_MEDICATIONS))
        for name in DEFAULT_MEDICATIONS:
            with self.subTest(name=name):
                self.assertRegex(name, r"^[a-z]+$")
        # Plain text also uses these names for lab tests, nutrients or body parts.
        for name in ("potassium", "iron", "magnesium", "oxygen", "thyroid"):
            with self.subTest(left_out=name):
                self.assertNotIn(name, DEFAULT_MEDICATIONS)

    def test_empty_medication_names_checks_numbers_only(self):
        spans = UnsupportedFactFilter(medication_names=[]).find(SUMMARY, CONTEXT)
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


SCORED = "Take lisinopril 10 mg daily. See your cardiologist in 2 weeks."


def make_span(text: str, label: str, summary: str = SCORED, after: int = 0) -> dict:
    """Builds a span for the first ``text`` in ``summary`` at or after ``after``."""
    start = summary.index(text, after)
    return {"start": start, "end": start + len(text), "text": text, "label": label}


class TestSpanScores(unittest.TestCase):
    def test_worked_example(self):
        pred = [
            make_span("lisinopril", "medication"),
            make_span("10", "number"),
            make_span("2", "number", after=SCORED.index("in 2")),
        ]
        gold = [
            make_span("lisinopril 10 mg", "medication_unsupported"),
            make_span("cardiologist", "name_unsupported"),
        ]
        scores = span_scores([SCORED], [pred], [gold])
        self.assertEqual(
            [scores[k] for k in ("correct", "partial", "spurious", "missed")],
            [0, 1, 2, 1],
        )
        self.assertAlmostEqual(scores["precision"], 0.5 / 3)
        self.assertAlmostEqual(scores["recall"], 0.5 / 2)
        self.assertAlmostEqual(scores["f1"], 0.2)

    def test_exact_match_ignores_labels(self):
        pred = [make_span("cardiologist", "medication")]
        gold = [make_span("cardiologist", "name_unsupported")]
        scores = span_scores([SCORED], [pred], [gold])
        self.assertEqual(scores["correct"], 1)
        self.assertEqual(
            [scores[k] for k in ("precision", "recall", "f1")], [1.0, 1.0, 1.0]
        )

    def test_neighbouring_words_count_as_overlap(self):
        pred = [make_span("10", "number")]
        gold = [make_span("mg", "word_unsupported")]
        self.assertEqual(span_scores([SCORED], [pred], [gold])["partial"], 1)

    def test_no_predictions_gives_zero_scores(self):
        gold = [make_span("cardiologist", "name_unsupported")]
        scores = span_scores([SCORED], [[]], [gold])
        self.assertEqual(scores["missed"], 1)
        self.assertEqual(
            [scores[k] for k in ("precision", "recall", "f1")], [0.0, 0.0, 0.0]
        )

    def test_counts_are_summed_over_summaries(self):
        exact = [make_span("cardiologist", "name_unsupported")]
        pred_b = [make_span("10", "number"), make_span("daily", "word")]
        gold_b = [make_span("weeks", "time_unsupported")]
        scores = span_scores([SCORED, SCORED], [exact, pred_b], [exact, gold_b])
        self.assertAlmostEqual(scores["precision"], 1 / 3)
        self.assertAlmostEqual(scores["recall"], 1 / 2)
        self.assertAlmostEqual(scores["f1"], 0.4)

    def test_different_lengths_raise(self):
        with self.assertRaises(ValueError):
            span_scores([SCORED], [[], []], [[]])


if __name__ == "__main__":
    unittest.main()
