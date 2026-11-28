import argparse
import csv
import itertools
import time
from pathlib import Path

from dotenv import load_dotenv

from ablation_settings import ERROR_TYPES_THE_FILTER_CANNOT_SEE, FILTER_SETTINGS
from dataset import APPROACHES, build_dataset
from errors import ALL_ERROR_TYPES, ERROR_TYPES_THE_FILTER_CAN_SEE
from filter_runs import summarize_filter_run

# Every value here is tried in every combination with every other.
ERROR_TYPE_SETS = {
    **{error_type: [error_type] for error_type in ALL_ERROR_TYPES},
    "visible_types": ERROR_TYPES_THE_FILTER_CAN_SEE,
    "invisible_types": ERROR_TYPES_THE_FILTER_CANNOT_SEE,
    "all_types": ALL_ERROR_TYPES,
}
ERROR_RATES = [0.1, 0.3, 0.5]
ERRORS_PER_SUMMARY_VALUES = [1, 2, 3]
HARMLESS_RATES = [0.0, 0.3, 0.6]

SETTING_COLUMNS = [
    "approach",
    "filter",
    "error_types",
    "error_rate",
    "errors_per_summary",
    "harmless_rate",
]
METRIC_COLUMNS = [
    "precision",
    "recall",
    "f1",
    "correct",
    "partial",
    "spurious",
    "missed",
    "error_sentences_removed",
    "correct_sentences_removed",
    "records",
    "records_with_planted_errors",
]


def all_data_settings(approach_names):
    data_settings_list = []
    for (
        approach_name,
        error_type_set_name,
        error_rate,
        errors_per_summary,
        harmless_rate,
    ) in itertools.product(
        approach_names,
        ERROR_TYPE_SETS,
        ERROR_RATES,
        ERRORS_PER_SUMMARY_VALUES,
        HARMLESS_RATES,
    ):
        error_types = ERROR_TYPE_SETS[error_type_set_name]
        if errors_per_summary > len(error_types):
            continue
        data_settings_list.append(
            {
                "approach_name": approach_name,
                "error_type_set_name": error_type_set_name,
                "error_types": error_types,
                "error_rate": error_rate,
                "errors_per_summary": errors_per_summary,
                "harmless_rate": harmless_rate,
            }
        )
    return data_settings_list


def results_row(data_settings, filter_name, summary):
    row = {
        "approach": data_settings["approach_name"],
        "filter": filter_name,
        "error_types": data_settings["error_type_set_name"],
        "error_rate": data_settings["error_rate"],
        "errors_per_summary": data_settings["errors_per_summary"],
        "harmless_rate": data_settings["harmless_rate"],
    }
    for metric_name in METRIC_COLUMNS:
        row[metric_name] = summary[metric_name]
    flagged_anything = summary["correct"] + summary["partial"] + summary["spurious"] > 0
    if not flagged_anything:
        row["precision"] = ""
    return row


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approaches", nargs="+", choices=list(APPROACHES), default=list(APPROACHES)
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/grid/grid_results.csv")
    return parser.parse_args()


def main():
    load_dotenv()
    arguments = read_arguments()
    data_settings_list = all_data_settings(arguments.approaches)
    total_runs = len(data_settings_list) * len(FILTER_SETTINGS)
    dataset_count = len(data_settings_list)
    print(
        f"{dataset_count} datasets x {len(FILTER_SETTINGS)} filters = {total_runs} runs"
    )

    results_rows = []
    start_time = time.time()
    for dataset_number, data_settings in enumerate(data_settings_list, start=1):
        settings = {**data_settings, "count": arguments.count, "seed": arguments.seed}
        records = build_dataset(settings)
        for filter_name, filter_checks in FILTER_SETTINGS.items():
            summary = summarize_filter_run(
                records,
                filter_checks,
                data_settings["approach_name"],
                data_settings["error_type_set_name"],
                filter_name,
            )
            results_rows.append(results_row(data_settings, filter_name, summary))
        if dataset_number % 25 == 0 or dataset_number == dataset_count:
            seconds = time.time() - start_time
            print(f"   {dataset_number} of {dataset_count} datasets ({seconds:.0f} s)")

    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=SETTING_COLUMNS + METRIC_COLUMNS
        )
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"Saved {len(results_rows)} runs to {output_path}")


if __name__ == "__main__":
    main()
