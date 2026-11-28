import csv
from pathlib import Path

from pyhealth.tasks.patient_summarization_filter import DEFAULT_MEDICATIONS

PROJECT_FOLDER = Path(__file__).parent


def load_diagnosis_drug_rows():
    with open(
        PROJECT_FOLDER / "diagnosis_drugs.csv", newline="", encoding="utf-8"
    ) as input_file:
        return list(csv.DictReader(input_file))


DIAGNOSIS_DRUG_ROWS = load_diagnosis_drug_rows()

DIAGNOSES = sorted({row["diagnosis"] for row in DIAGNOSIS_DRUG_ROWS})

DRUGS = sorted(DEFAULT_MEDICATIONS)

DOSES = [5, 10, 20, 25, 40, 50, 100, 250, 500, 750, 1000]

DAYS = [3, 5, 7, 10, 14]

HARMLESS_SENTENCES = [
    "Call 911 if you have trouble breathing or chest pain.",
    "Call 911 if you feel faint or confused.",
    "Return if your temperature is above 101 degrees.",
    "Drink 8 glasses of water a day.",
]
