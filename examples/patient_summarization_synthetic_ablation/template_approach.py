# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: The template approach: clean pairs built from sentence
#     templates filled with a diagnosis, drugs, a dose and a number of days.

import random

from vocabulary import DAYS, DIAGNOSIS_DRUG_ROWS, DRUGS


#this file builds the clean pairs using sentence templates and the word lists from vocabulary (this is the hybrid approach).
#this gives us an unlimited amount of data but its all written in the same style


#this file implements the template approach for data generation, which uses predefined phrases with keyword switching to generat ethe data.

ADMISSION_PHRASINGS = [
    "Patient was admitted with {diagnosis}.",
    "Patient presented to the hospital and was found to have {diagnosis}.",
    "Admitted for management of {diagnosis}.",
]
TREATMENT_PHRASINGS = [
    "Treated with {treatment_drug} {dose} mg daily for {days} days with improvement.",
    "Started on {treatment_drug} {dose} mg and completed a {days} day course.",
    "Received {treatment_drug} {dose} mg for {days} days and symptoms resolved.",
]
HOME_DRUG_PHRASINGS = [
    "Home {home_drug} was continued.",
    "Continued on home {home_drug}.",
    "went home when completed rec {home_drug}"
]

STOPPED_DRUG_PHRASINGS = [
    "Home {stopped_drug} was stopped during the admission.",
    "{stopped_drug} was held and discontinued at discharge.",
]

SUMMARY_DIAGNOSIS_PHRASINGS = [
    "You were in the hospital for {diagnosis}.",
    "patient had {diagnosis}",
    "You came to the hospital because of {diagnosis}.",
]

SUMMARY_TREATMENT_PHRASINGS = [
    "You were treated with {treatment_drug} {dose} mg for {days} days.",
    "We gave you {treatment_drug} {dose} mg for {days} days.",
]

SUMMARY_HOME_DRUG_PHRASINGS = [
    "Please keep taking your {home_drug}.",
    "Continue your {home_drug} at home.",
    "take {home_drug} going foward"
]

SUMMARY_STOPPED_DRUG_PHRASINGS = [
    "Do not take your {stopped_drug} until you see your doctor.",
    "We stopped your {stopped_drug}.",
]

SUMMARY_CLOSING_PHRASINGS = [
    "Follow up with your primary care doctor.",
    "see your doctor for further questions",
    "Please see your doctor at your next appointment.",
]


#this fills the sentence templates with a diagnosis, drugs and dose to make one clean pair
def create_clean_pair(random_generator: random.Random) -> dict:
    """Builds one clean pair from the templates."""
    diagnosis_drug_row = random_generator.choice(DIAGNOSIS_DRUG_ROWS)
    diagnosis = diagnosis_drug_row["diagnosis"]
    treatment_drug = diagnosis_drug_row["treatment_drug"]
    dose = diagnosis_drug_row["dose_mg"]
    other_drugs = [drug for drug in DRUGS if drug != treatment_drug]
    home_drug, stopped_drug = random_generator.sample(other_drugs, 2)
    days = random_generator.choice([day for day in DAYS if str(day) not in dose])

    slot_values = {
        "diagnosis": diagnosis,
        "treatment_drug": treatment_drug,
        "home_drug": home_drug,
        "stopped_drug": stopped_drug,
        "dose": dose,
        "days": days,
    }

    context_sentences = [
        random_generator.choice(ADMISSION_PHRASINGS),
        random_generator.choice(TREATMENT_PHRASINGS),
        random_generator.choice(HOME_DRUG_PHRASINGS),
        random_generator.choice(STOPPED_DRUG_PHRASINGS),
        "Patient was stable at discharge.",
    ]
    summary_sentences = [
        random_generator.choice(SUMMARY_DIAGNOSIS_PHRASINGS),
        random_generator.choice(SUMMARY_TREATMENT_PHRASINGS),
        random_generator.choice(SUMMARY_HOME_DRUG_PHRASINGS),
        random_generator.choice(SUMMARY_STOPPED_DRUG_PHRASINGS),
        random_generator.choice(SUMMARY_CLOSING_PHRASINGS),
    ]

    context = " ".join(context_sentences).format(**slot_values)
    summary = " ".join(summary_sentences).format(**slot_values)
    context = context[0].upper() + context[1:]


    return {
        "context": context,
        "summary": summary,
        "diagnosis": diagnosis,
        "treatment_drug": treatment_drug,
        "dose": dose,
        "stopped_drug": stopped_drug,
    }




#this repeats the template approach to make the requested number of pairs
def create_clean_pairs(count: int, random_generator: random.Random) -> list[dict]:
    """Builds ``count`` clean pairs from the templates."""
    return [create_clean_pair(random_generator) for _ in range(count)]
