# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Runs four small comparisons using the existing PyHealth filter.

from pyhealth.tasks.patient_summarization_filter import (
    UnsupportedFactFilter,
    span_scores,
)

from dataset import build_dataset
from errors import ALL_ERROR_TYPES, ERROR_TYPES_THE_FILTER_CAN_SEE


#this runs the filter and prints precision, recall and f1
def print_scores(name, examples, kinds=("number", "medication")):
    """Run the filter and print precision, recall and F1 as percentages."""
    fact_filter = UnsupportedFactFilter(kinds=kinds)
    summaries = []
    predictions = []
    labels = []
    for example in examples:
        summaries.append(example["summary"])
        predictions.append(fact_filter.find(example["summary"], example["context"]))
        labels.append(example["labels"])

    scores = span_scores(summaries, predictions, labels)
    print(
        f"{name}: "
        f"precision {100 * scores['precision']:.1f}%, "
        f"recall {100 * scores['recall']:.1f}%, "
        f"F1 {100 * scores['f1']:.1f}%"
    )


#this runs experiments a to d by changing one setting at a time
def run_all_experiments(settings):
    """Change one setting at a time, keeping the count and seed fixed."""
    print(f"\nExperiments for {settings['approach_name']}")

    # A: Try three filters on the same data.
    print("\nA. Filter checks")
    examples = build_dataset(settings)
    print_scores("Numbers only", examples, kinds=["number"])
    print_scores("Medications only", examples, kinds=["medication"])
    print_scores("Both checks", examples)

    # B: Compare detectable errors with the full set of five error types.
    print("\nB. Error mix (both checks)")
    detectable_settings = settings.copy()
    detectable_settings["error_types"] = ERROR_TYPES_THE_FILTER_CAN_SEE
    print_scores("Detectable types only", build_dataset(detectable_settings))
    all_types_settings = settings.copy()
    all_types_settings["error_types"] = ALL_ERROR_TYPES
    print_scores("All five types", build_dataset(all_types_settings))

    # C: Increase the chance of planting an error.
    print("\nC. Error rate (both checks)")
    for error_rate in [0.1, 0.3, 0.5]:
        changed_settings = settings.copy()
        changed_settings["error_rate"] = error_rate
        examples = build_dataset(changed_settings)
        print_scores(f"{error_rate:.0%} error rate", examples)

    # D: Count summaries changed by cleaning, with and without harmless numbers.
    print("\nD. Cleaning (both checks)")
    fact_filter = UnsupportedFactFilter()
    for harmless_rate in [0.0, 0.3]:
        changed_settings = settings.copy()
        changed_settings["harmless_rate"] = harmless_rate
        examples = build_dataset(changed_settings)
        changed_count = 0
        for example in examples:
            summary = example["summary"]
            flagged = fact_filter.find(summary, example["context"])
            cleaned = fact_filter.drop_sentences(summary, flagged)
            if cleaned != summary:
                changed_count += 1
        print(
            f"{harmless_rate:.0%} harmless-number rate: "
            f"cleaning changed {changed_count} of {len(examples)} summaries"
        )


#this compares the template, handwritten and llm data using the same settings
def compare_approaches(settings, approach_names):
    """Compare the three data sources using both checks."""
    print("\nComparison of approaches (both checks)")
    for approach_name in approach_names:
        changed_settings = settings.copy()
        changed_settings["approach_name"] = approach_name
        print_scores(approach_name, build_dataset(changed_settings))
