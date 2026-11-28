# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Word lists for the synthetic data: diagnoses with the drugs and
#     doses that treat them, drug names, doses and harmless sentences.

import csv
from pathlib import Path
from pyhealth.tasks.patient_summarization_filter import DEFAULT_MEDICATIONS


#the file loads diagnosis drugs csv, which allows us to pair drugs to the diagnosis for enhanced realism in the artifical summaries.
#purpose is to make the summaries we generate more plausible.


PROJECT_FOLDER = Path(__file__).parent


#this loads the diagnosis, drug and dose combinations from the csv
def load_diagnosis_drug_rows() -> list[dict]:
    """Reads ``diagnosis_drugs.csv``."""
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
    "Drink 8 glasses of water a day.",]
