# Author: Jonathan Temkin
# NetID: jtemkin3
# Paper: Hegselmann et al., "A Data-Centric Approach To Generate Faithful and
#     High Quality Patient Summaries with Large Language Models", CHIL 2024.
#     https://arxiv.org/abs/2402.15422
# Description: The LLM approach: clean pairs written by an LLM, saved in
#     llm_clean_pairs.json.

import json
import random


#this file loads the LLM data.
#these pairs were written by Claude in a chat and saved locally, so a key is not required.

from errors import check_clean_pair
from vocabulary import PROJECT_FOLDER

#uses the same standardized data format

LLM_PAIRS_FILE = PROJECT_FOLDER / "llm_clean_pairs.json"


#this loads the saved llm pairs, checks them and randomly picks the pairs to use
def create_clean_pairs(count: int, random_generator: random.Random) -> list[dict]:
    """Picks ``count`` LLM-written pairs at random, with repeats."""
    llm_pairs = json.loads(LLM_PAIRS_FILE.read_text(encoding="utf-8"))
    for pair in llm_pairs:
        check_clean_pair(pair)
    return [dict(random_generator.choice(llm_pairs)) for _ in range(count)]
