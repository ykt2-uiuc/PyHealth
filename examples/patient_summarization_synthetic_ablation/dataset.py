import random

import handwritten_approach
import llm_approach
import template_approach
from errors import ALL_ERROR_TYPES, build_example

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
    "errors_per_summary": 1,
    "harmless_rate": 0.3,
}


def choose_error_types(settings, random_generator):
    if random_generator.random() >= settings["error_rate"]:
        return []
    number_of_errors = min(settings["errors_per_summary"], len(settings["error_types"]))
    return random_generator.sample(settings["error_types"], number_of_errors)


def build_dataset(settings):
    random_generator = random.Random(settings["seed"])
    approach = APPROACHES[settings["approach_name"]]
    clean_pairs = approach.create_clean_pairs(settings["count"], random_generator)

    examples = []
    for example_number, clean_pair in enumerate(clean_pairs, start=1):
        error_types_to_plant = choose_error_types(settings, random_generator)
        add_harmless = random_generator.random() < settings["harmless_rate"]
        example = build_example(
            clean_pair, error_types_to_plant, add_harmless, random_generator
        )
        example["patient_id"] = f"synthetic-{example_number:04d}"
        example["note_id"] = f"synthetic-{example_number:04d}-discharge"
        examples.append(example)
    return examples
