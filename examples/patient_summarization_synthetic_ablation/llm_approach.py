import json
import os

from anthropic import Anthropic

from errors import problems_with_clean_pair
from vocabulary import DRUGS, PROJECT_FOLDER

SAVED_PAIRS_FILE = PROJECT_FOLDER / "llm_clean_pairs.json"
PAIRS_PER_REQUEST = 10

PROMPT = """Write {number_of_pairs} fictional hospital discharge examples
as a JSON array.
Do not copy or paraphrase any real patient record.

Each item has these keys:
- "context": a Brief Hospital Course written by a doctor, 3 to 5 sentences.
- "summary": Discharge Instructions for the patient in plain language, 3 to 5 sentences.
- "diagnosis": the main diagnosis, written exactly as it appears in both texts.
- "treatment_drug": one generic drug used to treat it, lowercase.
- "dose": the treatment drug's dose in mg, digits only, such as "500".
- "stopped_drug": a generic home drug that was stopped, lowercase.

Rules:
- Every drug, number and fact in the summary must also appear in the context.
- The diagnosis, treatment_drug and dose must each appear exactly once in the summary.
- The context must mention that the stopped_drug was stopped.
- Use only these drugs: {allowed_drugs}.

Reply with the JSON array only."""


def ask_llm_for_pairs(number_of_pairs, allowed_drugs):
    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    prompt = PROMPT.format(
        number_of_pairs=number_of_pairs, allowed_drugs=", ".join(allowed_drugs)
    )
    response = client.messages.create(
        model=os.environ["LLM_MODEL"],
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    reply_text = response.content[0].text
    first_bracket = reply_text.index("[")
    last_bracket = reply_text.rindex("]")
    return json.loads(reply_text[first_bracket : last_bracket + 1])


def is_usable_pair(pair):
    return problems_with_clean_pair(pair) == []


def load_saved_pairs():
    if not SAVED_PAIRS_FILE.exists():
        return []
    return json.loads(SAVED_PAIRS_FILE.read_text(encoding="utf-8"))


def save_pairs(pairs):
    SAVED_PAIRS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SAVED_PAIRS_FILE.write_text(json.dumps(pairs, indent=2), encoding="utf-8")


def get_unique_pairs(number_wanted, allowed_drugs):
    pairs = load_saved_pairs()
    while len(pairs) < number_wanted:
        new_pairs = ask_llm_for_pairs(PAIRS_PER_REQUEST, allowed_drugs)
        usable_pairs = [pair for pair in new_pairs if is_usable_pair(pair)]
        pairs.extend(usable_pairs)
        save_pairs(pairs)
        print(f"LLM pairs: {len(pairs)} of {number_wanted}")
    return pairs[:number_wanted]


def create_clean_pairs(count, random_generator, unique_pair_count=12):
    unique_pairs = get_unique_pairs(unique_pair_count, DRUGS)
    return [dict(random_generator.choice(unique_pairs)) for _ in range(count)]
