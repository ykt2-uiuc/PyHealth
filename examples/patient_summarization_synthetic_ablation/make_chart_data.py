import csv
import json
from pathlib import Path

PROJECT_FOLDER = Path(__file__).parent
GRID_RESULTS_FILE = PROJECT_FOLDER / "outputs" / "grid" / "grid_results.csv"
CHART_DATA_FILE = Path(__file__).parent / "chart_data.json"

FILTERS = ["numbers_only", "medications_only", "numbers_and_medications"]
ERROR_TYPES = [
    "wrong_drug",
    "added_drug",
    "wrong_number",
    "wrong_diagnosis",
    "negation",
]
HARMLESS_RATES = ["0.0", "0.3", "0.6"]
APPROACHES = ["template", "handwritten", "llm"]


def average_percent(values):
    numbers = [float(value) for value in values if value != ""]
    if not numbers:
        return 0.0
    return round(100 * sum(numbers) / len(numbers), 1)


def main():
    with open(GRID_RESULTS_FILE, newline="", encoding="utf-8") as input_file:
        runs = list(csv.DictReader(input_file))

    recall_by_type = {
        filter_name: [
            average_percent(
                run["recall"]
                for run in runs
                if run["filter"] == filter_name and run["error_types"] == error_type
            )
            for error_type in ERROR_TYPES
        ]
        for filter_name in FILTERS
    }
    precision_by_harmless = {
        filter_name: [
            average_percent(
                run["precision"]
                for run in runs
                if run["filter"] == filter_name
                and run["harmless_rate"] == harmless_rate
            )
            for harmless_rate in HARMLESS_RATES
        ]
        for filter_name in FILTERS
    }
    both_checks_all_types = [
        run
        for run in runs
        if run["filter"] == "numbers_and_medications"
        and run["error_types"] == "all_types"
    ]
    cleaning_removed = [
        average_percent(
            run["error_sentences_removed"]
            for run in both_checks_all_types
            if run["harmless_rate"] == harmless_rate
        )
        for harmless_rate in HARMLESS_RATES
    ]
    cleaning_lost = [
        average_percent(
            run["correct_sentences_removed"]
            for run in both_checks_all_types
            if run["harmless_rate"] == harmless_rate
        )
        for harmless_rate in HARMLESS_RATES
    ]
    approach_f1 = {
        approach: average_percent(
            run["f1"] for run in both_checks_all_types if run["approach"] == approach
        )
        for approach in APPROACHES
    }

    chart_data = {
        "recall_by_type": recall_by_type,
        "precision_by_harmless": precision_by_harmless,
        "cleaning_removed": cleaning_removed,
        "cleaning_lost": cleaning_lost,
        "approach_f1": approach_f1,
        "runs": len(runs),
    }
    CHART_DATA_FILE.write_text(json.dumps(chart_data, indent=1), encoding="utf-8")
    print(f"Saved chart data from {len(runs)} runs to {CHART_DATA_FILE}")


if __name__ == "__main__":
    main()
