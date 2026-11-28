import argparse

from dotenv import load_dotenv

from dataset import APPROACHES, DEFAULT_SETTINGS, build_dataset
from errors import ALL_ERROR_TYPES
from experiments import compare_approaches, run_all_experiments
from file_formats import OUTPUT_FORMATS, save_dataset
from report import print_dataset_report, print_example_summaries


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--approach",
        choices=list(APPROACHES) + ["all"],
        default=DEFAULT_SETTINGS["approach_name"],
    )
    parser.add_argument("--count", type=int, default=DEFAULT_SETTINGS["count"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS["seed"])
    parser.add_argument(
        "--error-rate", type=float, default=DEFAULT_SETTINGS["error_rate"]
    )
    parser.add_argument(
        "--error-types",
        nargs="+",
        choices=ALL_ERROR_TYPES,
        default=DEFAULT_SETTINGS["error_types"],
    )
    parser.add_argument(
        "--errors-per-summary",
        type=int,
        default=DEFAULT_SETTINGS["errors_per_summary"],
    )
    parser.add_argument(
        "--harmless-rate", type=float, default=DEFAULT_SETTINGS["harmless_rate"]
    )
    parser.add_argument(
        "--formats", nargs="+", choices=list(OUTPUT_FORMATS), default=["jsonl"]
    )
    parser.add_argument("--output-folder", default="data")
    parser.add_argument("--file-name", default="examples")
    parser.add_argument("--show-examples", type=int, default=3)
    parser.add_argument("--skip-experiments", action="store_true")
    return parser.parse_args()


def check_settings(settings):
    if not 0 <= settings["error_rate"] <= 1:
        raise ValueError("--error-rate must be between 0 and 1")
    if not 0 <= settings["harmless_rate"] <= 1:
        raise ValueError("--harmless-rate must be between 0 and 1")
    if settings["errors_per_summary"] < 1:
        raise ValueError("--errors-per-summary must be at least 1")


def main():
    load_dotenv()
    arguments = read_arguments()

    settings = {
        "approach_name": arguments.approach,
        "count": arguments.count,
        "seed": arguments.seed,
        "error_rate": arguments.error_rate,
        "error_types": arguments.error_types,
        "errors_per_summary": arguments.errors_per_summary,
        "harmless_rate": arguments.harmless_rate,
    }
    check_settings(settings)

    if arguments.approach == "all":
        approach_names = list(APPROACHES)
    else:
        approach_names = [arguments.approach]

    for approach_name in approach_names:
        approach_settings = {**settings, "approach_name": approach_name}
        examples = build_dataset(approach_settings)
        file_name = arguments.file_name
        if arguments.approach == "all":
            file_name = f"{arguments.file_name}_{approach_name}"
        saved_paths = save_dataset(
            examples, arguments.output_folder, file_name, arguments.formats
        )
        print(f"\n=== {approach_name} ===")
        for saved_path in saved_paths:
            print(f"Saved {len(examples)} summaries to {saved_path}")
        print_dataset_report(examples)
        print_example_summaries(examples, arguments.show_examples)

    if arguments.skip_experiments:
        return
    if arguments.approach == "all":
        compare_approaches(settings, approach_names)
    else:
        run_all_experiments(settings)


if __name__ == "__main__":
    main()
