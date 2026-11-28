import csv
import json
from pathlib import Path

FIELD_ORDER = [
    "patient_id",
    "note_id",
    "context",
    "summary",
    "has_planted_errors",
    "mistakes",
    "error_types",
    "harmless_sentence",
    "labels",
]


def ordered_record(example):
    return {field_name: example[field_name] for field_name in FIELD_ORDER}


def filter_eval_record(example):
    return {
        "text": example["context"],
        "summary": example["summary"],
        "labels": example["labels"],
    }


def save_jsonl(examples, file_path):
    with open(file_path, "w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(ordered_record(example)) + "\n")


def save_json(examples, file_path):
    records = [ordered_record(example) for example in examples]
    Path(file_path).write_text(json.dumps(records, indent=2), encoding="utf-8")


def save_csv(examples, file_path):
    with open(file_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=FIELD_ORDER)
        writer.writeheader()
        for example in examples:
            row = ordered_record(example)
            row["mistakes"] = " | ".join(example["mistakes"])
            row["error_types"] = ";".join(example["error_types"])
            row["harmless_sentence"] = example["harmless_sentence"] or ""
            row["labels"] = json.dumps(example["labels"])
            writer.writerow(row)


def save_filter_eval_jsonl(examples, file_path):
    with open(file_path, "w", encoding="utf-8") as output_file:
        for example in examples:
            output_file.write(json.dumps(filter_eval_record(example)) + "\n")


OUTPUT_FORMATS = {
    "jsonl": (save_jsonl, ".jsonl"),
    "json": (save_json, ".json"),
    "csv": (save_csv, ".csv"),
    "filter-eval": (save_filter_eval_jsonl, ".filter_eval.jsonl"),
}


def save_dataset(examples, output_folder, file_name, format_names):
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for format_name in format_names:
        save_function, file_ending = OUTPUT_FORMATS[format_name]
        file_path = output_folder / (file_name + file_ending)
        save_function(examples, file_path)
        saved_paths.append(file_path)
    return saved_paths
