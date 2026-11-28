# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Plants a known error in clean (context, summary) pairs and
#     records where it is, so the filter can be scored against it.

import random
import re

from vocabulary import DIAGNOSES, DOSES, DRUGS, HARMLESS_SENTENCES


#this is the main file for planting the errors into the clean summaries (wrong drug, added drug, wrong number, wrong diagnosis, negation)
#it also records exactly where each error is so we can score the filter against it later.
#It also checks that the clean pairs are valid before they are used


# this takes the generated data (either llm generated, human generated, or hybird approach)
# and plants errors in the data. The loaders first check that each phrase to replace appears once in the summary.
# There are 5 types of errors that we can introduct into the data with this script


ALL_ERROR_TYPES = [
    "wrong_drug",
    "added_drug",
    "wrong_number",
    "wrong_diagnosis",
    "negation",
]

#5 error types above that can be added into the data

ERROR_TYPES_THE_FILTER_CAN_SEE = ["wrong_drug", "added_drug", "wrong_number"]


#this finds where a word or phrase appears and checks it only appears once
def find_exactly_once(text: str, phrase: str) -> int:
    """Finds the one place a phrase appears in a text, as a whole word."""
    whole_word_pattern = r"\b" + re.escape(phrase) + r"\b"
    matches = list(re.finditer(whole_word_pattern, text, flags=re.IGNORECASE))
    if len(matches) == 0:
        raise ValueError(f"'{phrase}' is missing from: {text}")
    if len(matches) > 1:
        raise ValueError(f"'{phrase}' appears more than once in: {text}")
    return matches[0].start()

#this performs validation to ensure the data that reaches this point is actually valid so we know where/how to impose the error

#this checks the diagnosis, drug and dose are in the pair before adding errors
def check_clean_pair(pair: dict) -> None:
    """Stops the run if an error can't be planted in a clean pair."""
    for field_name in ["diagnosis", "treatment_drug", "dose"]:
        if pair[field_name].lower() not in pair["context"].lower():
            raise ValueError(f"'{pair[field_name]}' is missing from: {pair['context']}")
        find_exactly_once(pair["summary"], pair[field_name])


#this records the error text and where it starts and ends in the summary
def make_label(
    start_position: int,
    text: str,
    error_type: str,
    original_text: str | None,
    added_sentence: str | None,
) -> dict:
    """Builds the answer-key label for a planted error."""
    return {
        "start": start_position,
        "end": start_position + len(text),
        "text": text,
        "label": error_type,
        "original_text": original_text,
        "added_sentence": added_sentence,
    }


#this describes what was changed or added to the summary
def describe_mistake(label: dict) -> str:
    """Describes a planted error in words."""
    if label["original_text"] is not None:
        return (
            f"{label['label']}: '{label['original_text']}' changed to '{label['text']}'"
        )
    return f"{label['label']}: added '{label['added_sentence']}'"


#this replaces a word or phrase and records the error position
def replace_phrase(
    summary: str, old_phrase: str, new_phrase: str, error_type: str
) -> tuple[str, dict]:
    """Replaces a phrase in the summary and labels the replacement as an error."""
    start_position = find_exactly_once(summary, old_phrase)
    end_position = start_position + len(old_phrase)
    original_text = summary[start_position:end_position]
    new_summary = summary[:start_position] + new_phrase + summary[end_position:]
    label = make_label(start_position, new_phrase, error_type, original_text, None)
    return new_summary, label


#this adds a sentence and records where the error is in it
def append_sentence(
    summary: str, sentence: str, phrase_to_label: str, error_type: str
) -> tuple[str, dict]:
    """Adds a sentence to the end of the summary and labels part of it."""
    new_summary = summary + " " + sentence
    start_position = len(summary) + 1 + sentence.index(phrase_to_label)
    label = make_label(start_position, phrase_to_label, error_type, None, sentence)
    return new_summary, label


#this gets the numbers that appear in the text
def numbers_in(text: str) -> set[str]:
    """Returns every run of digits in the text, such as ``{"500", "7"}``."""
    return set(re.findall(r"\d+", text))


#this picks a drug that is not already in the text
def pick_drug_not_in(text: str, random_generator: random.Random) -> str:
    """Picks a random drug from ``DRUGS`` that the text does not mention."""
    lowercase_text = text.lower()
    unused_drugs = [drug for drug in DRUGS if drug not in lowercase_text]
    return random_generator.choice(unused_drugs)

#this picks a dose whose number is not already in the text
def pick_dose_not_in(text: str, random_generator: random.Random) -> str:
    """Picks a random dose from ``DOSES`` whose number is not in the text."""
    text_numbers = numbers_in(text)
    unused_doses = [dose for dose in DOSES if str(dose) not in text_numbers]
    return str(random_generator.choice(unused_doses))


#this picks a diagnosis that is not already in the text
def pick_diagnosis_not_in(text: str, random_generator: random.Random) -> str:
    """Picks a random diagnosis from ``DIAGNOSES`` that the text does not mention."""
    lowercase_text = text.lower()
    unused_diagnoses = [
        diagnosis for diagnosis in DIAGNOSES if diagnosis.lower() not in lowercase_text
    ]
    return random_generator.choice(unused_diagnoses)


#this adds one of the 5 error types to a clean summary
def plant_error(
    clean_pair: dict, error_type: str, random_generator: random.Random
) -> tuple[str, dict]:
    """Plants one error of the given type in the clean summary."""
    summary = clean_pair["summary"]
    context_and_summary = clean_pair["context"] + " " + summary
    #introduct a drug mismatch error
    if error_type == "wrong_drug":
        new_drug = pick_drug_not_in(context_and_summary, random_generator)
        old_drug = clean_pair["treatment_drug"]
        return replace_phrase(summary, old_drug, new_drug, error_type)
    #introduct a mismatching #
    if error_type == "wrong_number":
        new_dose = pick_dose_not_in(context_and_summary, random_generator)
        old_dose = clean_pair["dose"]
        return replace_phrase(summary, old_dose, new_dose, error_type)
    #change the diagnosis
    if error_type == "wrong_diagnosis":
        new_diagnosis = pick_diagnosis_not_in(context_and_summary, random_generator)
        old_diagnosis = clean_pair["diagnosis"]
        return replace_phrase(summary, old_diagnosis, new_diagnosis, error_type)

    #add a new drug
    if error_type == "added_drug":
        new_drug = pick_drug_not_in(context_and_summary, random_generator)
        sentence = f"Start taking {new_drug} every day."
        return append_sentence(summary, sentence, new_drug, error_type)

    #the summary tells the patient to keep taking a drug that was actually stopped
    if error_type == "negation":
        sentence = f"Keep taking your {clean_pair['stopped_drug']} as before."
        return append_sentence(summary, sentence, "Keep taking", error_type)

    raise ValueError(f"unknown error type: {error_type}")

#builds the final example with error introduced
def build_example(
    clean_pair: dict,
    error_type: str | None,
    add_harmless: bool,
    random_generator: random.Random,
) -> dict:
    """Turns a clean pair into one dataset record with its answer key."""
    summary = clean_pair["summary"]
    labels = []
    if error_type is not None:
        summary, label = plant_error(clean_pair, error_type, random_generator)
        labels.append(label)
    harmless_sentence = None
    if add_harmless:
        harmless_sentence = random_generator.choice(HARMLESS_SENTENCES)
        summary = summary + " " + harmless_sentence
    return {
        "context": clean_pair["context"],
        "summary": summary,
        "error_type": error_type,
        "mistake": describe_mistake(labels[0]) if labels else None,
        "labels": labels,
        "harmless_sentence": harmless_sentence,
    }
