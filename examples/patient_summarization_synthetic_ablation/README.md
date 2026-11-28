# DL4H synthetic data for the unsupported-fact filter

Builds synthetic (context, summary) pairs, plants one known error in selected
summaries, and measures how well the existing PyHealth filter finds it.

## Setup and run

Use Python 3.12 or 3.13 with PyHealth installed. From the repository root:

    pip install -e .
    cd examples/patient_summarization_synthetic_ablation
    python main.py --approach template

Choose `template`, `handwritten`, `llm`, or `all`:

    python main.py --approach all --count 500 --seed 0

Each chosen approach saves its dataset, prints the first summary and its
planted mistake (or `None` for a clean summary), and runs Experiments A to D. `all` also prints a
side-by-side comparison of the three approaches. No API key is needed: the
LLM approach only reads the saved pairs in `llm_clean_pairs.json`.

There are four command-line options:

| Option | Default | Meaning |
|---|---|---|
| `--approach` | `template` | Which source of clean pairs to use |
| `--count` | `500` | Number of summaries; must be positive |
| `--seed` | `0` | Random seed for repeatable results |
| `--preset` | `all` | Every bundle, or one named bundle |

Edit `DEFAULT_SETTINGS` in `dataset.py` to change the error rate (default
30%), error types (all five), or harmless-number rate (default 30%). Rates
are probabilities, so the actual share varies slightly. Each selected
summary gets exactly one error.

## Settings bundles

The bundles are in `PRESETS` in `dataset.py`. Each starts with
`DEFAULT_SETTINGS` and changes only the listed values. These are the bundles
with the supplied defaults:

| Preset | Error rate | Error types | Harmless-number rate |
|---|---|---|---|
| `default` | 30% | All five | 30% |
| `low_error` | 10% | All five | 30% |
| `high_error` | 50% | All five | 30% |
| `detectable_only` | 30% | Wrong drug, added drug, wrong number | 30% |
| `meaning_only` | 30% | Wrong diagnosis, negation | 30% |
| `no_harmless` | 30% | All five | 0% |
| `many_harmless` | 30% | All five | 60% |
| `clean_control` | 0% | None planted | 0% |

For example:

    python main.py --approach all --preset all
    python main.py --approach all --preset low_error
    python main.py --approach all --preset high_error
    python main.py --approach all --preset meaning_only

By default the script runs every bundle, printing A-D and an approach
comparison for each. To save all the output in PowerShell:

    python main.py --approach all | Tee-Object -FilePath results_all_bundles.txt

Use `--preset default` to run only the original baseline settings.

Use the same count and seed to compare bundles. The selected preset controls
the saved datasets, Experiment A and the final approach comparison. B still
compares its two error mixes, C still compares its three error rates, and D
still compares its two harmless-number rates. Those experiments override
that one setting while keeping the rest of the preset. For example, C plants
errors even when the selected preset is `clean_control`.

## Experiments

All experiments use the same dataset size and seed. A compares filters on
one dataset; B to D rebuild data with the changed setting.

| Experiment | What changes |
|---|---|
| A | Numbers only, medications only, or both checks |
| B | Only detectable error types versus all five types |
| C | Error rate: 10%, 30%, 50% |
| D | Harmless-number rate: 0%, 30%; counts summaries changed by sentence removal |

The five planted error types are `wrong_drug`, `added_drug`, `wrong_number`,
`wrong_diagnosis`, and `negation`. Negation adds an instruction to keep taking
a drug that the context says was stopped. Harmless sentences include advice
such as calling 911, which can trigger the number check even though it is
reasonable advice.

A to C and the approach comparison print precision, recall and F1 as
percentages using the existing `span_scores` function. D prints how many
summaries changed after `drop_sentences` removed flagged sentences. That count
includes both planted errors and false alarms; it does not measure whether
each removal was correct. These experiments measure a rule-based filter on
synthetic examples; they do not train a model.

## Saved output

Files go under `data/` relative to the directory where you run the script:

    data/default/examples_template.jsonl
    data/default/examples_handwritten.jsonl
    data/default/examples_llm.jsonl

Other presets use their own folder, such as `data/low_error/`, so bundles do
not overwrite each other's data. Only the chosen approaches are saved.
Running the same preset again replaces its files. Older files directly under
`data/` are from earlier runs and are not updated.
Each JSONL line contains `patient_id`, `note_id`, `context`, `summary`,
`error_type`, `mistake`, `labels`, and `harmless_sentence`. A clean example has
`error_type` and `mistake` set to `null`, and an empty `labels` list.

Each planted error has one label with `start`, `end`, `text`, `label`,
`original_text`, and `added_sentence`. Positions are character offsets into
`summary`, with `end` exclusive. `original_text` records a replacement;
`added_sentence` records an addition. The unused field is `null`.

Results are printed. To save the whole printout:

    python main.py --approach all > results.txt

## Results

Default run: 500 summaries per approach, seed 0, error rate 30%, harmless
numbers 30%, both checks:

| Approach | Precision | Recall | F1 |
|---|---|---|---|
| template | 38.2 | 59.3 | 46.5 |
| handwritten | 40.2 | 60.4 | 48.2 |
| llm | 37.9 | 60.9 | 46.7 |

In this run, the filter finds every `wrong_drug`, `added_drug`, and
`wrong_number` error and misses every `wrong_diagnosis` and `negation` error.
Harmless numbers cause false alarms and lower precision.

## Input files

| File | Used by |
|---|---|
| `diagnosis_drugs.csv` | Template approach: diagnosis, treatment drug and dose |
| `handwritten_pairs.csv` | Handwritten approach: 10 saved clean pairs |
| `llm_clean_pairs.json` | LLM approach: 12 saved clean pairs written by Claude in a chat |

Both saved-pair files use `context`, `summary`, `diagnosis`, `treatment_drug`,
`dose`, and `stopped_drug`. The two saved-pair approaches sample with
replacement; increasing `--count` repeats these pairs rather than producing
new writing.

A small check requires diagnosis, treatment drug and dose to appear in the
context and exactly once in the summary as whole words, ignoring case. Keep
all summary facts supported by the context, and make sure the context says
that `stopped_drug` was stopped. Write doses with a space before the unit
(`500 mg`).

## Code layout

| File | What it does |
|---|---|
| `main.py` | Reads options, saves JSONL, prints examples and runs experiments |
| `dataset.py` | Holds defaults, seeds randomness and selects summaries to change |
| `errors.py` | Plants one error and records its exact position |
| `template_approach.py` | Builds clean pairs from sentence templates |
| `handwritten_approach.py` | Loads the handwritten CSV and samples clean pairs |
| `llm_approach.py` | Loads the saved LLM JSON and samples clean pairs |
| `vocabulary.py` | Loads diagnosis/drug rows and defines doses and harmless advice |
| `experiments.py` | Calls the existing filter, scores predictions and prints A to D |

The filter and scoring function are imported from
`pyhealth/tasks/patient_summarization_filter.py`.
