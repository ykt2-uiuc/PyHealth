# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: The handwritten approach: clean pairs written by hand in
#     handwritten_pairs.csv.

import csv
import random

#this file implements the handwritten data which are stored in a local csv file


from errors import check_clean_pair
from vocabulary import PROJECT_FOLDER


#this file loads the handwritten pairs from handwritten_pairs.csv and checks each row is valid before using it.
#if a row is broken the run stops and tells you which row and what the problem is


HANDWRITTEN_PAIRS_FILE = PROJECT_FOLDER / "handwritten_pairs.csv"


#this loads the handwritten pairs from the csv and checks each pair
def load_handwritten_pairs() -> list[dict]:
    """Reads ``handwritten_pairs.csv`` and checks every row."""
    with open(HANDWRITTEN_PAIRS_FILE, newline="", encoding="utf-8") as input_file:
        pairs = list(csv.DictReader(input_file))
    for row_number, pair in enumerate(pairs, start=2):
        try:
            check_clean_pair(pair)
        except ValueError as problem:
            raise ValueError(f"handwritten_pairs.csv row {row_number}: {problem}")
    return pairs


#this randomly picks the requested number of handwritten pairs, with repeats allowed
def create_clean_pairs(count: int, random_generator: random.Random) -> list[dict]:
    """Picks ``count`` handwritten pairs at random, with repeats."""
    handwritten_pairs = load_handwritten_pairs()
    return [dict(random_generator.choice(handwritten_pairs)) for _ in range(count)]
