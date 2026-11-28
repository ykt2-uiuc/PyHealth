# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Makes synthetic pairs, saves JSONL, and runs Experiments A to D.
# Usage: python main.py --approach all --count 500 --seed 0

import argparse
import json
from pathlib import Path

from dataset import APPROACHES, DEFAULT_SETTINGS, PRESETS, build_dataset
from experiments import compare_approaches, run_all_experiments


#this reads the approach, preset, count and seed supplied when running the script
def read_arguments() -> argparse.Namespace:
    """Reads the approach, settings bundle, dataset size and random seed."""
    parser = argparse.ArgumentParser(description="Generate data and run Experiments A to D.")
    parser.add_argument(
        "--approach", choices=list(APPROACHES) + ["all"],
        default=DEFAULT_SETTINGS["approach_name"],
    )
    parser.add_argument("--count", type=int, default=DEFAULT_SETTINGS["count"])
    parser.add_argument("--seed", type=int, default=DEFAULT_SETTINGS["seed"])
    parser.add_argument("--preset", choices=list(PRESETS) + ["all"], default="all")
    arguments = parser.parse_args()
    if arguments.count < 1:
        parser.error("--count must be at least 1")
    return arguments


#this builds and saves the data then runs the experiments for the selected approaches
def main() -> None:
    """Runs the same pipeline for each selected bundle of settings."""
    arguments = read_arguments()
    preset_names = [arguments.preset]
    if arguments.preset == "all":
        preset_names = PRESETS

    approach_names = [arguments.approach]
    if arguments.approach == "all":
        approach_names = APPROACHES

    #this repeats the same steps for each bundle of settings
    for preset_name in preset_names:
        settings = DEFAULT_SETTINGS.copy()
        settings.update(PRESETS[preset_name])
        settings["count"] = arguments.count
        settings["seed"] = arguments.seed
        output_folder = Path("data") / preset_name
        output_folder.mkdir(parents=True, exist_ok=True)
        print(f"\nPreset: {preset_name}, count {arguments.count}, seed {arguments.seed}")
        print(
            f"Error rate: {settings['error_rate']:.0%}, "
            f"harmless-number rate: {settings['harmless_rate']:.0%}"
        )
        print(f"Error types: {', '.join(settings['error_types'])}")

        for approach_name in approach_names:
            settings["approach_name"] = approach_name
            examples = build_dataset(settings)
            output_path = output_folder / f"examples_{approach_name}.jsonl"
            with output_path.open("w", encoding="utf-8") as output_file:
                for example in examples:
                    output_file.write(json.dumps(example) + "\n")
            print(f"\n=== {approach_name} ===")
            print(f"Saved {len(examples)} summaries to {output_path}")
            print(f"Example summary: {examples[0]['summary']}")
            print(f"Planted mistake: {examples[0]['mistake']}")
            run_all_experiments(settings)

        if arguments.approach == "all":
            compare_approaches(settings, approach_names)


if __name__ == "__main__":
    main()
