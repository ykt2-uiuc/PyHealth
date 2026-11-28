from pyhealth.tasks.patient_summarization_filter import (
    UnsupportedFactFilter,
    span_scores,
)

from dataset import build_dataset
from errors import ALL_ERROR_TYPES, ERROR_TYPES_THE_FILTER_CAN_SEE

FILTER_SETTINGS = {
    "numbers only": ["number"],
    "medications only": ["medication"],
    "both": ["number", "medication"],
}


def score(examples, fact_filter):
    summaries = [example["summary"] for example in examples]
    predicted_spans = [
        fact_filter.find(example["summary"], example["context"]) for example in examples
    ]
    gold_spans = [example["labels"] for example in examples]
    return span_scores(summaries, predicted_spans, gold_spans)


def spans_overlap(first_span, second_span):
    return (
        first_span["start"] < second_span["end"]
        and second_span["start"] < first_span["end"]
    )


def found_by_error_type(examples, fact_filter):
    found_counts = {error_type: 0 for error_type in ALL_ERROR_TYPES}
    planted_counts = {error_type: 0 for error_type in ALL_ERROR_TYPES}
    for example in examples:
        predicted_spans = fact_filter.find(example["summary"], example["context"])
        for gold_span in example["labels"]:
            error_type = gold_span["label"]
            planted_counts[error_type] += 1
            if any(
                spans_overlap(gold_span, predicted) for predicted in predicted_spans
            ):
                found_counts[error_type] += 1
    return found_counts, planted_counts


def split_into_sentences(text):
    sentences = []
    start_position = 0
    for position, character in enumerate(text):
        is_sentence_end = character in ".!?"
        next_is_space_or_end = position + 1 == len(text) or text[position + 1] == " "
        if is_sentence_end and next_is_space_or_end:
            sentences.append((start_position, position + 1))
            start_position = position + 2
    return sentences


def cleaning_effect(examples, fact_filter):
    sentences_with_errors = 0
    sentences_with_errors_removed = 0
    correct_sentences = 0
    correct_sentences_removed = 0
    for example in examples:
        summary = example["summary"]
        predicted_spans = fact_filter.find(summary, example["context"])
        cleaned_summary = fact_filter.drop_sentences(summary, predicted_spans)
        for start_position, end_position in split_into_sentences(summary):
            sentence = {"start": start_position, "end": end_position}
            sentence_text = summary[start_position:end_position]
            was_removed = sentence_text not in cleaned_summary
            has_error = any(
                spans_overlap(sentence, label) for label in example["labels"]
            )
            if has_error:
                sentences_with_errors += 1
                sentences_with_errors_removed += was_removed
            else:
                correct_sentences += 1
                correct_sentences_removed += was_removed
    share_of_error_sentences_removed = sentences_with_errors_removed / max(
        sentences_with_errors, 1
    )
    share_of_correct_sentences_removed = correct_sentences_removed / max(
        correct_sentences, 1
    )
    return {
        "share_of_error_sentences_removed": share_of_error_sentences_removed,
        "share_of_correct_sentences_removed": share_of_correct_sentences_removed,
        "sentences_with_errors": sentences_with_errors,
        "correct_sentences": correct_sentences,
    }


def print_table_header(title):
    print()
    print(title)
    print(
        f"{'setting':<26}{'P':>7}{'R':>7}{'F1':>7}{'correct':>9}{'partial':>9}{'spurious':>9}{'missed':>9}"
    )


def print_table_row(setting_name, scores):
    print(
        f"{setting_name:<26}"
        f"{100 * scores['precision']:>7.1f}"
        f"{100 * scores['recall']:>7.1f}"
        f"{100 * scores['f1']:>7.1f}"
        f"{scores['correct']:>9}"
        f"{scores['partial']:>9}"
        f"{scores['spurious']:>9}"
        f"{scores['missed']:>9}"
    )


def experiment_filter_settings(settings):
    examples = build_dataset(settings)
    print_table_header("A. Filter settings")
    for setting_name, kinds in FILTER_SETTINGS.items():
        fact_filter = UnsupportedFactFilter(kinds=kinds)
        print_table_row(setting_name, score(examples, fact_filter))

    found_counts, planted_counts = found_by_error_type(
        examples, UnsupportedFactFilter()
    )
    print()
    print("   Planted errors found, by type (both checks)")
    for error_type in ALL_ERROR_TYPES:
        found = found_counts[error_type]
        planted = planted_counts[error_type]
        if planted == 0:
            continue
        percent = 100 * found / planted
        print(f"   {error_type:<18}{found:>4} of {planted:<4} ({percent:.0f}%)")


def experiment_error_mix(settings):
    both_checks = UnsupportedFactFilter()
    print_table_header("B. Error mix (both checks)")
    visible_settings = {**settings, "error_types": ERROR_TYPES_THE_FILTER_CAN_SEE}
    all_types_settings = {**settings, "error_types": ALL_ERROR_TYPES}
    print_table_row(
        "visible types only", score(build_dataset(visible_settings), both_checks)
    )
    print_table_row(
        "all five types", score(build_dataset(all_types_settings), both_checks)
    )


def experiment_error_rate(settings):
    both_checks = UnsupportedFactFilter()
    print_table_header("C. Error rate (both checks)")
    for error_rate in [0.1, 0.3, 0.5]:
        rate_settings = {**settings, "error_rate": error_rate}
        examples = build_dataset(rate_settings)
        print_table_row(f"{error_rate:.0%} of summaries", score(examples, both_checks))


def experiment_cleaning(settings):
    both_checks = UnsupportedFactFilter()
    print()
    print("D. Cleaning with drop_sentences (both checks)")
    for harmless_rate in [0.0, 0.3]:
        harmless_settings = {**settings, "harmless_rate": harmless_rate}
        effect = cleaning_effect(build_dataset(harmless_settings), both_checks)
        print(
            f"   harmless numbers in {harmless_rate:.0%} of summaries: "
            f"removed {100 * effect['share_of_error_sentences_removed']:.0f}% "
            f"of {effect['sentences_with_errors']} sentences with errors and "
            f"{100 * effect['share_of_correct_sentences_removed']:.1f}% "
            f"of {effect['correct_sentences']} correct sentences"
        )


def run_all_experiments(settings):
    print(
        f"\nExperiments: {settings['approach_name']} approach, "
        f"{settings['count']} summaries, seed {settings['seed']}, "
        f"error rate {settings['error_rate']:.0%}, "
        f"up to {settings['errors_per_summary']} error(s) per summary, "
        f"harmless numbers {settings['harmless_rate']:.0%}"
    )
    print(f"   error types: {', '.join(settings['error_types'])}")
    experiment_filter_settings(settings)
    experiment_error_mix(settings)
    experiment_error_rate(settings)
    experiment_cleaning(settings)


def compare_approaches(settings, approach_names):
    both_checks = UnsupportedFactFilter()
    print(
        f"\nComparison of approaches: {settings['count']} summaries each, "
        f"seed {settings['seed']}, error rate {settings['error_rate']:.0%}, "
        f"harmless numbers {settings['harmless_rate']:.0%}"
    )
    print_table_header("Both checks, all chosen error types")
    results_by_approach = {}
    for approach_name in approach_names:
        approach_settings = {**settings, "approach_name": approach_name}
        examples = build_dataset(approach_settings)
        print_table_row(approach_name, score(examples, both_checks))
        found_counts, planted_counts = found_by_error_type(examples, both_checks)
        results_by_approach[approach_name] = (found_counts, planted_counts)

    print()
    print("Planted errors found, by type")
    print(f"   {'error type':<18}" + "".join(f"{name:>14}" for name in approach_names))
    for error_type in settings["error_types"]:
        row = f"   {error_type:<18}"
        for approach_name in approach_names:
            found_counts, planted_counts = results_by_approach[approach_name]
            found = found_counts[error_type]
            planted = planted_counts[error_type]
            row += f"{f'{found}/{planted}':>14}"
        print(row)

    print()
    print("Cleaning with drop_sentences")
    for approach_name in approach_names:
        approach_settings = {**settings, "approach_name": approach_name}
        effect = cleaning_effect(build_dataset(approach_settings), both_checks)
        error_percent = 100 * effect["share_of_error_sentences_removed"]
        correct_percent = 100 * effect["share_of_correct_sentences_removed"]
        print(
            f"   {approach_name:<12} removed {error_percent:.0f}% of sentences "
            f"with errors and {correct_percent:.1f}% of correct sentences"
        )
