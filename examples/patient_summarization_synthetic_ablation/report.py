from collections import Counter


def mark_errors(summary, labels):
    marked_summary = summary
    for label in sorted(labels, key=lambda label: label["start"], reverse=True):
        start_position = label["start"]
        end_position = label["end"]
        marked_summary = (
            marked_summary[:start_position]
            + "[["
            + marked_summary[start_position:end_position]
            + "]]"
            + marked_summary[end_position:]
        )
    return marked_summary


def print_dataset_report(examples):
    summaries_with_errors = [example for example in examples if example["labels"]]
    summaries_with_harmless = [
        example for example in examples if example["harmless_sentence"]
    ]
    error_type_counts = Counter(
        label["label"] for example in examples for label in example["labels"]
    )
    errors_per_summary_counts = Counter(len(example["labels"]) for example in examples)

    total = len(examples)
    print(f"\nDataset report: {total} summaries")
    print(
        f"   with at least one error: {len(summaries_with_errors)} "
        f"({100 * len(summaries_with_errors) / total:.0f}%)"
    )
    print(
        f"   with a harmless number:  {len(summaries_with_harmless)} "
        f"({100 * len(summaries_with_harmless) / total:.0f}%)"
    )
    print("   errors planted, by type:")
    for error_type, count in sorted(error_type_counts.items()):
        print(f"      {error_type:<18}{count}")
    print("   summaries by number of errors:")
    for number_of_errors, count in sorted(errors_per_summary_counts.items()):
        print(f"      {number_of_errors} errors: {count}")


def print_example_summaries(examples, how_many):
    examples_with_errors = [example for example in examples if example["labels"]]
    for example in examples_with_errors[:how_many]:
        print(
            f"\n   {example['patient_id']}  errors: {', '.join(example['error_types'])}"
        )
        print(f"   context: {example['context']}")
        print(f"   summary: {mark_errors(example['summary'], example['labels'])}")
        for mistake in example["mistakes"]:
            print(f"   mistake: {mistake}")
