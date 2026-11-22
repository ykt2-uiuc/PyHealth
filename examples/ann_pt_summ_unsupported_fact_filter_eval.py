# Author: Leo
# NetID: lab8
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: Scores the unsupported-fact filter against the expert-labeled
#     hallucinations in the ann-pt-summ dataset on PhysioNet.

import json
from pathlib import Path


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
