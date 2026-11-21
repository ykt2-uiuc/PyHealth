# Author: Leo
# NetID: lab8
# Description: Fast tests for the unsupported-fact filter using only made-up
#     (synthetic) text. No real MIMIC data is used.

import unittest

from pyhealth.tasks.patient_summarization_filter import UnsupportedFactFilter

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


if __name__ == "__main__":
    unittest.main()
