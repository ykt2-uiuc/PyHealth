import argparse
from pathlib import Path

from dotenv import load_dotenv

from ablation_settings import DATA_SCENARIOS, FILTER_SETTINGS, scenario_settings
from dataset import APPROACHES, build_dataset
from filter_runs import (
    run_filter_on_records,
    save_filtered_records,
    save_results_table,
    summarize_filter_run,
)


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approaches", nargs="+", choices=list(APPROACHES), default=list(APPROACHES)
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(DATA_SCENARIOS),
        default=list(DATA_SCENARIOS),
    )
    parser.add_argument(
        "--filters",
        nargs="+",
        choices=list(FILTER_SETTINGS),
        default=list(FILTER_SETTINGS),
    )
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-folder", default="outputs/ablation")
    return parser.parse_args()


def print_results_row(results_row):
    print(
        f"{results_row['approach']:<12}"
        f"{results_row['scenario']:<26}"
        f"{results_row['filter']:<25}"
        f"{100 * results_row['precision']:>7.1f}"
        f"{100 * results_row['recall']:>7.1f}"
        f"{100 * results_row['f1']:>7.1f}"
        f"{100 * results_row['error_sentences_removed']:>9.0f}"
        f"{100 * results_row['correct_sentences_removed']:>9.1f}"
    )


def main():
    load_dotenv()
    arguments = read_arguments()
    output_folder = Path(arguments.output_folder)

    print(
        f"{'approach':<12}{'scenario':<26}{'filter':<25}"
        f"{'P':>7}{'R':>7}{'F1':>7}{'err rm%':>9}{'ok rm%':>9}"
    )
    results_rows = []
    for approach_name in arguments.approaches:
        for scenario_name in arguments.scenarios:
            settings = {
                **scenario_settings(scenario_name),
                "approach_name": approach_name,
                "count": arguments.count,
                "seed": arguments.seed,
            }
            records = build_dataset(settings)
            for filter_name in arguments.filters:
                filter_checks = FILTER_SETTINGS[filter_name]
                filtered_records = run_filter_on_records(
                    records, filter_checks, filter_name
                )
                file_path = (
                    output_folder
                    / approach_name
                    / scenario_name
                    / f"{filter_name}.jsonl"
                )
                save_filtered_records(filtered_records, file_path)
                results_row = summarize_filter_run(
                    records, filter_checks, approach_name, scenario_name, filter_name
                )
                results_rows.append(results_row)
                print_results_row(results_row)

    results_path = output_folder / "results.csv"
    save_results_table(results_rows, results_path)
    print(f"\nSaved {len(results_rows)} runs to {output_folder}")
    print(f"Results table: {results_path}")


if __name__ == "__main__":
    main()
