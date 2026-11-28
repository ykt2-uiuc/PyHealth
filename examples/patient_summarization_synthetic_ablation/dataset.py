# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Builds a synthetic dataset from one of the three approaches,
#     planting errors and harmless numbers at the chosen rates.

import random

import handwritten_approach
import llm_approach
import template_approach
from errors import ALL_ERROR_TYPES, build_example


#THIS FILE TAKES THE CLEAN PAIRS FROM WHICHEVER APPROACH IS SELECTED (TEMPLATE, HANDWRITTEN, OR LLM) AND TURNS THEM INTO THE FINAL DATASET.
#it decides which summaries get errors and harmless numbers based on the settings, and the seed makes sure the same settings always give the same data.


#this is the consolidated function that builds the data per the user specification and ensures the errors are implanted per the user instructions and the specified error rate.
# it can leverage any of the 3 data generation approaches using the below, with the default settings specified.

APPROACHES = {
    "template": template_approach,
    "handwritten": handwritten_approach,
    "llm": llm_approach,
}

DEFAULT_SETTINGS = {
    "approach_name": "template",
    "count": 500,
    "seed": 0,
    "error_rate": 0.3,
    "error_types": ALL_ERROR_TYPES,
    "harmless_rate": 0.3,
}

#these bundles only change the settings listed, everything else uses the defaults
PRESETS = {
    "default": {},
    "low_error": {"error_rate": 0.1},
    "high_error": {"error_rate": 0.5},
    #these separate errors the filter can detect from errors of meaning
    "detectable_only": {
        "error_types": ["wrong_drug", "added_drug", "wrong_number"],
    },
    "meaning_only": {"error_types": ["wrong_diagnosis", "negation"]},
    "no_harmless": {"harmless_rate": 0.0},
    "many_harmless": {"harmless_rate": 0.6},
    #this is a clean control with no planted errors or added harmless numbers
    "clean_control": {"error_rate": 0.0, "harmless_rate": 0.0},
}


#this builds the dataset and picks which summaries get errors using the seed
def build_dataset(settings: dict) -> list[dict]:
    """Builds a synthetic dataset with planted errors and their answer key."""
    random_generator = random.Random(settings["seed"])
    approach = APPROACHES[settings["approach_name"]]
    clean_pairs = approach.create_clean_pairs(settings["count"], random_generator)

    examples = []
    for example_number, clean_pair in enumerate(clean_pairs, start=1):
        error_type = None
        if random_generator.random() < settings["error_rate"]:
            error_type = random_generator.choice(settings["error_types"])
        add_harmless = random_generator.random() < settings["harmless_rate"]
        example = build_example(clean_pair, error_type, add_harmless, random_generator)
        example["patient_id"] = f"synthetic-{example_number:04d}"
        example["note_id"] = f"synthetic-{example_number:04d}-discharge"
        examples.append(example)
    return examples
