import csv

from errors import problems_with_clean_pair
from vocabulary import PROJECT_FOLDER

HANDWRITTEN_PAIRS_FILE = PROJECT_FOLDER / "handwritten_pairs.csv"


def load_handwritten_pairs():
    with open(HANDWRITTEN_PAIRS_FILE, newline="", encoding="utf-8") as input_file:
        pairs = list(csv.DictReader(input_file))
    for row_number, pair in enumerate(pairs, start=2):
        problems = problems_with_clean_pair(pair)
        if problems:
            raise ValueError(
                f"{HANDWRITTEN_PAIRS_FILE.name} row {row_number}: "
                + "; ".join(problems)
            )
    return pairs


def create_clean_pairs(count, random_generator):
    handwritten_pairs = load_handwritten_pairs()
    return [dict(random_generator.choice(handwritten_pairs)) for _ in range(count)]
