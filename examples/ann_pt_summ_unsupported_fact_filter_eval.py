# Author: Leo
# NetID: lab8
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Scores the unsupported-fact filter against the expert-labeled
#     hallucinations in the ann-pt-summ dataset on PhysioNet. Prints only
#     totals, never text from the data.
#
# Usage:
#     python examples/ann_pt_summ_unsupported_fact_filter_eval.py \
#         --data <labels.jsonl> [--medications <names.txt>]

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pyhealth.tasks.patient_summarization_filter import (
    UnsupportedFactFilter,
    span_scores,
)

# The filter settings to compare: each name maps to the kinds it checks.
SETTINGS = {
    "number": ("number",),
    "medication": ("medication",),
    "both": ("number", "medication"),
}

COUNTS = ("correct", "partial", "spurious", "missed")


def load_examples(path: str | Path) -> list[dict]:
    """Reads an expert-labeled file: one JSON object per line.

    Each object has ``text`` (the Brief Hospital Course), ``summary`` (the
    Discharge Instructions) and ``labels``: the expert spans, as character
    offsets into ``summary``.

    Args:
        path: The JSONL file.

    Returns:
        The examples in file order.
    """
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def evaluate(examples: list[dict], fact_filter: UnsupportedFactFilter) -> dict:
    """Runs the filter on each example and scores its spans against the experts.

    Args:
        examples: Examples from :func:`load_examples`.
        fact_filter: The filter to score.

    Returns:
        The scores from ``span_scores``: precision, recall, f1 and the counts.
    """
    summaries = [ex["summary"] for ex in examples]
    pred = [fact_filter.find(ex["summary"], ex["text"]) for ex in examples]
    gold = [ex["labels"] for ex in examples]
    return span_scores(summaries, pred, gold)


def read_medication_names(path: str | Path) -> list[str]:
    """Reads medication names, one per line, and skips blank lines."""
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    """Prints precision, recall and F1 (in %) and the counts per filter setting.

    Without ``--medications``, only the numbers are checked.

    Args:
        argv: Command-line arguments. If ``None``, reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(
        description="Score the unsupported-fact filter against expert labels."
    )
    parser.add_argument("--data", required=True, help="expert-labeled JSONL file")
    parser.add_argument("--medications", help="text file, one name per line")
    args = parser.parse_args(argv)

    examples = load_examples(args.data)
    names = read_medication_names(args.medications) if args.medications else []
    settings = SETTINGS if names else {"number": SETTINGS["number"]}

    header = (f"{c:>9}" for c in COUNTS)
    print(f"{'filter':<12}{'P':>7}{'R':>7}{'F1':>7}", *header, sep="")
    for setting, kinds in settings.items():
        fact_filter = UnsupportedFactFilter(kinds=kinds, medication_names=names)
        scores = evaluate(examples, fact_filter)
        percents = (f"{100 * scores[k]:>7.1f}" for k in ("precision", "recall", "f1"))
        counts = (f"{scores[c]:>9}" for c in COUNTS)
        print(f"{setting:<12}", *percents, *counts, sep="")


if __name__ == "__main__":
    main()
