import argparse
import csv
import json
from pathlib import Path

from grid_search import METRIC_COLUMNS

TEMPLATE_FILE = Path(__file__).parent / "dashboard_template.html"
SHARE_COLUMNS = ["error_rate", "harmless_rate"]
COUNT_COLUMNS = ["errors_per_summary"]
NUMBER_COLUMNS = METRIC_COLUMNS + SHARE_COLUMNS + COUNT_COLUMNS


def read_grid_results(file_path):
    with open(file_path, newline="", encoding="utf-8") as input_file:
        rows = list(csv.DictReader(input_file))
    for row in rows:
        for column in NUMBER_COLUMNS:
            if row[column] == "":
                row[column] = None
                continue
            row[column] = float(row[column])
            if row[column].is_integer() and column not in SHARE_COLUMNS:
                row[column] = int(row[column])
    return rows


def read_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="outputs/grid/grid_results.csv")
    parser.add_argument("--output", default="outputs/grid/dashboard.html")
    return parser.parse_args()


def main():
    arguments = read_arguments()
    runs = read_grid_results(arguments.results)
    template = TEMPLATE_FILE.read_text(encoding="utf-8")
    page = template.replace("__RUNS_DATA__", json.dumps(runs))
    output_path = Path(arguments.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    print(f"Saved a dashboard of {len(runs)} runs to {output_path}")


if __name__ == "__main__":
    main()
