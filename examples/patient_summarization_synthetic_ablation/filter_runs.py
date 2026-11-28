import csv
import json
from pathlib import Path

from pyhealth.tasks.patient_summarization_filter import (
    UnsupportedFactFilter,
    span_scores,
)

from experiments import cleaning_effect

RESULTS_COLUMNS = [
    "approach",
    "scenario",
    "filter",
    "records",
    "records_with_planted_errors",
    "precision",
    "recall",
    "f1",
    "correct",
    "partial",
    "spurious",
    "missed",
    "error_sentences_removed",
    "correct_sentences_removed",
]


def run_filter_on_record(record, fact_filter, filter_name):
    flagged_spans = fact_filter.find(record["summary"], record["context"])
    cleaned_summary = fact_filter.drop_sentences(record["summary"], flagged_spans)
    record_scores = span_scores(
        [record["summary"]], [flagged_spans], [record["labels"]]
    )
    filtered_record = {
        **record,
        "filter": filter_name,
        "flagged_spans": flagged_spans,
        "flagged_texts": [span["text"] for span in flagged_spans],
        "cleaned_summary": cleaned_summary,
        "correctly_flagged": record_scores["correct"],
        "partly_flagged": record_scores["partial"],
        "false_alarms": record_scores["spurious"],
        "missed_errors": record_scores["missed"],
    }
    filtered_record["filter_caught_every_error"] = filtered_record["missed_errors"] == 0
    return filtered_record


def run_filter_on_records(records, filter_checks, filter_name):
    fact_filter = UnsupportedFactFilter(kinds=filter_checks)
    return [
        run_filter_on_record(record, fact_filter, filter_name) for record in records
    ]


def summarize_filter_run(
    records, filter_checks, approach_name, scenario_name, filter_name
):
    fact_filter = UnsupportedFactFilter(kinds=filter_checks)
    all_flagged_spans = [
        fact_filter.find(record["summary"], record["context"]) for record in records
    ]
    scores = span_scores(
        [record["summary"] for record in records],
        all_flagged_spans,
        [record["labels"] for record in records],
    )
    effect = cleaning_effect(records, fact_filter)
    return {
        "approach": approach_name,
        "scenario": scenario_name,
        "filter": filter_name,
        "records": len(records),
        "records_with_planted_errors": sum(
            record["has_planted_errors"] for record in records
        ),
        "precision": round(scores["precision"], 4),
        "recall": round(scores["recall"], 4),
        "f1": round(scores["f1"], 4),
        "correct": scores["correct"],
        "partial": scores["partial"],
        "spurious": scores["spurious"],
        "missed": scores["missed"],
        "error_sentences_removed": round(effect["share_of_error_sentences_removed"], 4),
        "correct_sentences_removed": round(
            effect["share_of_correct_sentences_removed"], 4
        ),
    }


def save_filtered_records(filtered_records, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as output_file:
        for filtered_record in filtered_records:
            output_file.write(json.dumps(filtered_record) + "\n")


def save_results_table(results_rows, file_path):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        writer.writerows(results_rows)
