# DL4H synthetic data for the unsupported-fact filter

Makes fake (context, summary) pairs, plants errors in the summaries at known
positions, and scores the PyHealth unsupported-fact filter against them.

## Setup

    cd examples/patient_summarization_synthetic_ablation
    pip install -r requirements.txt

For the `llm` approach, copy `env_example.txt` to a new file named `.env` and add your key:

    ANTHROPIC_API_KEY=your-key
    LLM_MODEL=claude-sonnet-5-5

## Run

    python main.py --approach template
    python main.py --approach handwritten
    python main.py --approach llm
    python main.py --approach all

`--approach all` runs the three approaches with the same settings and prints a
side-by-side comparison.

Each run saves the data, prints a report of what was planted with example
summaries (errors marked like [[this]]), then runs Experiments A to D.

## Options

| Option | Default | What it controls |
|---|---|---|
| `--approach` | `template` | `template`, `handwritten`, `llm` or `all` |
| `--count` | `500` | Number of summaries |
| `--seed` | `0` | Random seed, so a run can be repeated |
| `--error-rate` | `0.3` | Share of summaries that get errors |
| `--error-types` | all five | Any of `wrong_drug added_drug wrong_number wrong_diagnosis negation` |
| `--errors-per-summary` | `1` | Errors planted in each summary that gets errors |
| `--harmless-rate` | `0.3` | Share of summaries that get a harmless number (such as 911) |
| `--formats` | `jsonl` | Any of `jsonl json csv filter-eval` |
| `--output-folder` | `data` | Where files are saved |
| `--file-name` | `examples` | File name before the ending |
| `--show-examples` | `3` | Example summaries printed with errors marked |
| `--skip-experiments` | off | Only make and save the data |

Examples:

    python main.py --error-types wrong_drug negation --error-rate 0.5
    python main.py --errors-per-summary 3 --error-rate 1.0 --show-examples 5
    python main.py --formats jsonl csv filter-eval --skip-experiments

## Output formats

| Format | File | Fields |
|---|---|---|
| `jsonl` | `examples.jsonl` | One record per line: `patient_id`, `note_id`, `context`, `summary`, `has_planted_errors`, `mistakes`, `error_types`, `harmless_sentence`, `labels` |
| `json` | `examples.json` | The same records as one list |
| `csv` | `examples.csv` | The same fields; `mistakes` joined by ` \| `, `error_types` by `;`, and `labels` as JSON text |
| `filter-eval` | `examples.filter_eval.jsonl` | `text`, `summary`, `labels`: the input of `examples/ann_pt_summ_unsupported_fact_filter_eval.py` |

`has_planted_errors` is true when the summary was changed on purpose, and false
for a clean summary. `mistakes` lists each planted error in words, for example
`wrong_drug: 'azithromycin' changed to 'amoxicillin'` or
`added_drug: added 'Start taking aspirin every day.'`.

`patient_id`, `note_id`, `context` and `summary` match the samples that the
`PatientSummarizationMIMIC4` task makes. Each label is
`{"start", "end", "text", "label", "original_text", "added_sentence"}`, with
character positions into `summary`. `original_text` is the text that was
replaced, or `null` when a sentence was added; `added_sentence` is the added
sentence, or `null` when text was replaced.

## Ablation

    python run_ablation.py

Runs every combination of data approach, data scenario and filter setting,
saves every record with the filter's output, and writes one results table.
Narrow it down with `--approaches`, `--scenarios`, `--filters`, `--count`
(default 200), `--seed` and `--output-folder` (default `outputs/ablation`):

    python run_ablation.py --approaches handwritten --scenarios baseline high_error_rate
    python run_ablation.py --filters numbers_only medications_only --count 500

Scenarios and filter settings are named in `ablation_settings.py`. Each
scenario changes one thing from the baseline (30% error rate, all five error
types, 1 error per summary, harmless numbers in 30%):

| Scenario | Change from baseline |
|---|---|
| `baseline` | none |
| `visible_errors_only` | only `wrong_drug`, `added_drug`, `wrong_number` |
| `invisible_errors_only` | only `wrong_diagnosis`, `negation` |
| `low_error_rate` | error rate 10% |
| `high_error_rate` | error rate 50% |
| `no_harmless_numbers` | harmless numbers 0% |
| `many_harmless_numbers` | harmless numbers 60% |
| `three_errors_per_summary` | 3 errors in each summary that gets errors |

| Filter setting | Checks |
|---|---|
| `no_filter` | none (flags nothing; the reference point) |
| `numbers_only` | numbers |
| `medications_only` | medication names |
| `numbers_and_medications` | both |

Output:

    outputs/ablation/results.csv
    outputs/ablation/<approach>/<scenario>/<filter>.jsonl

Each `.jsonl` record holds the data record's fields plus `filter`,
`flagged_spans`, `flagged_texts`, `cleaned_summary` (the summary after
`drop_sentences`), `correctly_flagged`, `partly_flagged`, `false_alarms`,
`missed_errors` and `filter_caught_every_error`.

`results.csv` has one row per run: precision, recall, F1, the counts, and the
share of sentences with errors and of correct sentences that cleaning removed.

## Grid search and comparison page

    python grid_search.py
    python make_dashboard.py

`grid_search.py` tries every combination of the values at the top of the file
and saves one row per run to `outputs/grid/grid_results.csv` (about 3 to 4
minutes with the defaults):

| Setting | Values |
|---|---|
| approach | `template`, `handwritten`, `llm` |
| error types | each single type, `visible_types`, `invisible_types`, `all_types` |
| error rate | 10%, 30%, 50% |
| errors per summary | 1, 2, 3 (only up to the number of types in the set) |
| harmless rate | 0%, 30%, 60% |
| filter | `no_filter`, `numbers_only`, `medications_only`, `numbers_and_medications` |

That is 351 datasets x 4 filters = 1,404 runs of 200 summaries each. Use
`--count`, `--seed` and `--approaches` to change it. Precision is left blank
when the filter flagged nothing, since there is nothing to be precise about.

`make_dashboard.py` turns that file into one page,
`outputs/grid/dashboard.html`, that opens in any browser with no internet
connection. It has a filter for every setting, a heatmap whose rows, columns
and metric you choose, the 10 best runs for the chosen metric, and a sortable
table of every matching run. The page layout is in `dashboard_template.html`.

## Results and presentation

These files hold the outputs of a full run, so they can be read without
rerunning anything:

| File | What it is |
|---|---|
| `grid_results.csv` | All 1,404 grid search runs |
| `dashboard.html` | The comparison page for those runs |
| `ablation_results.csv` | The 96 ablation runs |
| `approach_comparison.txt` | Output of `python main.py --approach all` |

`dl4h_ablation_presentation.pptx` is a 12-slide deck for the
video, with speaker notes on every slide. Its shapes, tables and charts are
native PowerPoint objects, so text and chart data can be edited directly
(right-click a chart and choose Edit Data). To rebuild it after rerunning the
grid search:

    python make_chart_data.py
    npm install
    npm run build

## Input files

All data the code reads is in these files, in this folder. Edit them to change the data.

| File | Columns | Used by |
|---|---|---|
| `diagnosis_drugs.csv` | `diagnosis`, `treatment_drug`, `dose_mg` | `template`: each row is a diagnosis with a drug and dose that treat it |
| `handwritten_pairs.csv` | `context`, `summary`, `diagnosis`, `treatment_drug`, `dose`, `stopped_drug` | `handwritten` |
| `llm_clean_pairs.json` | the same fields as `handwritten_pairs.csv` | `llm`: the saved pairs; the API is called only if there are fewer than 12 |

Rules for a pair, checked when it loads (a broken row stops the run and names
the row and the problem):

- `diagnosis`, `treatment_drug`, `dose` and `stopped_drug` appear in the context.
- `diagnosis`, `treatment_drug` and `dose` each appear exactly once in the
  summary, as a whole word.
- Every number and listed drug in the summary also appears in the context, or
  the filter will flag the clean pair.

Matching ignores capitalization. Write doses with a space before the unit
("500 mg", not "500mg").

The pairs in `llm_clean_pairs.json` were written by Claude in a chat, not
through the API.

## Files

| File | What it does |
|---|---|
| `main.py` | Reads the options, makes and saves the data, prints the report, runs the experiments |
| `template_approach.py` | Builds pairs from sentence templates and word lists |
| `handwritten_approach.py` | Loads and checks `handwritten_pairs.csv`, then picks from it |
| `llm_approach.py` | Loads `llm_clean_pairs.json`, asking the LLM for more pairs if needed |
| `errors.py` | Plants errors in a clean pair and records where they are |
| `vocabulary.py` | Loads `diagnosis_drugs.csv`; drugs, doses and harmless sentences |
| `dataset.py` | Turns clean pairs into a dataset with the chosen settings |
| `file_formats.py` | Saves the dataset in each format |
| `report.py` | Prints what was planted and example summaries |
| `ablation_settings.py` | Names the data scenarios and filter settings for the ablation |
| `filter_runs.py` | Runs a filter on every record, scores it, and saves the outputs |
| `run_ablation.py` | Runs every approach, scenario and filter combination |
| `grid_search.py` | Runs every combination of settings and saves the metrics |
| `make_dashboard.py` | Builds the comparison page from the grid search results |
| `dashboard_template.html` | The comparison page's layout and code |
| `make_chart_data.py` | Turns the grid results into the numbers the deck's charts use |
| `build_presentation.js` | Builds the deck |
| `experiments.py` | Experiments A to D, the comparison of approaches, and the printed tables |
