import re

from vocabulary import DIAGNOSES, DOSES, DRUGS, HARMLESS_SENTENCES

ALL_ERROR_TYPES = [
    "wrong_drug",
    "added_drug",
    "wrong_number",
    "wrong_diagnosis",
    "negation",
]

ERROR_TYPES_THE_FILTER_CAN_SEE = ["wrong_drug", "added_drug", "wrong_number"]


def find_exactly_once(text, phrase):
    whole_word_pattern = r"\b" + re.escape(phrase) + r"\b"
    matches = list(re.finditer(whole_word_pattern, text, flags=re.IGNORECASE))
    if len(matches) == 0:
        raise ValueError(f"'{phrase}' is missing from: {text}")
    if len(matches) > 1:
        raise ValueError(f"'{phrase}' appears more than once in: {text}")
    return matches[0].start()


REQUIRED_PAIR_FIELDS = [
    "context",
    "summary",
    "diagnosis",
    "treatment_drug",
    "dose",
    "stopped_drug",
]


def problems_with_clean_pair(pair):
    problems = []
    for field_name in REQUIRED_PAIR_FIELDS:
        if not pair.get(field_name):
            problems.append(f"'{field_name}' is empty or missing")
    if problems:
        return problems
    for field_name in ["diagnosis", "treatment_drug", "dose", "stopped_drug"]:
        if pair[field_name].lower() not in pair["context"].lower():
            problems.append(f"{field_name} '{pair[field_name]}' is not in the context")
    for field_name in ["diagnosis", "treatment_drug", "dose"]:
        try:
            find_exactly_once(pair["summary"], pair[field_name])
        except ValueError:
            problems.append(
                f"{field_name} '{pair[field_name]}' must appear exactly once "
                "in the summary as a whole word"
            )
    return problems


def make_label(start_position, text, error_type, original_text, added_sentence):
    return {
        "start": start_position,
        "end": start_position + len(text),
        "text": text,
        "label": error_type,
        "original_text": original_text,
        "added_sentence": added_sentence,
    }


def describe_mistake(label):
    if label["original_text"] is not None:
        return (
            f"{label['label']}: '{label['original_text']}' changed to '{label['text']}'"
        )
    return f"{label['label']}: added '{label['added_sentence']}'"


def shift_labels_after(labels, position, length_change):
    for label in labels:
        if label["start"] > position:
            label["start"] += length_change
            label["end"] += length_change


def replace_phrase(summary, labels, old_phrase, new_phrase, error_type):
    start_position = find_exactly_once(summary, old_phrase)
    end_position = start_position + len(old_phrase)
    original_text = summary[start_position:end_position]
    new_summary = summary[:start_position] + new_phrase + summary[end_position:]
    shift_labels_after(labels, start_position, len(new_phrase) - len(old_phrase))
    labels.append(
        make_label(start_position, new_phrase, error_type, original_text, None)
    )
    return new_summary


def append_sentence(summary, labels, sentence, phrase_to_label, error_type):
    new_summary = summary + " " + sentence
    start_position = len(summary) + 1 + sentence.index(phrase_to_label)
    labels.append(
        make_label(start_position, phrase_to_label, error_type, None, sentence)
    )
    return new_summary


def numbers_in(text):
    return set(re.findall(r"\d+", text))


def pick_drug_not_in(text, random_generator):
    lowercase_text = text.lower()
    unused_drugs = [drug for drug in DRUGS if drug not in lowercase_text]
    return random_generator.choice(unused_drugs)


def pick_dose_not_in(text, random_generator):
    text_numbers = numbers_in(text)
    unused_doses = [dose for dose in DOSES if str(dose) not in text_numbers]
    return str(random_generator.choice(unused_doses))


def pick_diagnosis_not_in(text, random_generator):
    lowercase_text = text.lower()
    unused_diagnoses = [
        diagnosis for diagnosis in DIAGNOSES if diagnosis.lower() not in lowercase_text
    ]
    return random_generator.choice(unused_diagnoses)


def plant_error(clean_pair, summary, labels, error_type, random_generator):
    context = clean_pair["context"]
    context_and_summary = context + " " + summary

    if error_type == "wrong_drug":
        new_drug = pick_drug_not_in(context_and_summary, random_generator)
        old_drug = clean_pair["treatment_drug"]
        return replace_phrase(summary, labels, old_drug, new_drug, error_type)

    if error_type == "wrong_number":
        new_dose = pick_dose_not_in(context_and_summary, random_generator)
        old_dose = clean_pair["dose"]
        return replace_phrase(summary, labels, old_dose, new_dose, error_type)

    if error_type == "wrong_diagnosis":
        new_diagnosis = pick_diagnosis_not_in(context_and_summary, random_generator)
        old_diagnosis = clean_pair["diagnosis"]
        return replace_phrase(summary, labels, old_diagnosis, new_diagnosis, error_type)

    if error_type == "added_drug":
        new_drug = pick_drug_not_in(context_and_summary, random_generator)
        sentence = f"Start taking {new_drug} every day."
        return append_sentence(summary, labels, sentence, new_drug, error_type)

    if error_type == "negation":
        sentence = f"Keep taking your {clean_pair['stopped_drug']} as before."
        return append_sentence(summary, labels, sentence, "Keep taking", error_type)

    raise ValueError(f"unknown error type: {error_type}")


def build_example(clean_pair, error_types_to_plant, add_harmless, random_generator):
    summary = clean_pair["summary"]
    labels = []
    for error_type in error_types_to_plant:
        summary = plant_error(clean_pair, summary, labels, error_type, random_generator)
    harmless_sentence = None
    if add_harmless:
        harmless_sentence = random_generator.choice(HARMLESS_SENTENCES)
        summary = summary + " " + harmless_sentence
    labels.sort(key=lambda label: label["start"])
    return {
        "context": clean_pair["context"],
        "summary": summary,
        "has_planted_errors": len(labels) > 0,
        "mistakes": [describe_mistake(label) for label in labels],
        "labels": labels,
        "error_types": list(error_types_to_plant),
        "harmless_sentence": harmless_sentence,
    }
