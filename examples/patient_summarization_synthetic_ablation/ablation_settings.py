from errors import ALL_ERROR_TYPES, ERROR_TYPES_THE_FILTER_CAN_SEE

ERROR_TYPES_THE_FILTER_CANNOT_SEE = [
    error_type
    for error_type in ALL_ERROR_TYPES
    if error_type not in ERROR_TYPES_THE_FILTER_CAN_SEE
]

BASELINE_SCENARIO = {
    "error_rate": 0.3,
    "error_types": ALL_ERROR_TYPES,
    "errors_per_summary": 1,
    "harmless_rate": 0.3,
}

# Each scenario changes one thing from the baseline.
DATA_SCENARIOS = {
    "baseline": {},
    "visible_errors_only": {"error_types": ERROR_TYPES_THE_FILTER_CAN_SEE},
    "invisible_errors_only": {"error_types": ERROR_TYPES_THE_FILTER_CANNOT_SEE},
    "low_error_rate": {"error_rate": 0.1},
    "high_error_rate": {"error_rate": 0.5},
    "no_harmless_numbers": {"harmless_rate": 0.0},
    "many_harmless_numbers": {"harmless_rate": 0.6},
    "three_errors_per_summary": {"errors_per_summary": 3},
}

# The value is the list of checks the filter runs.
FILTER_SETTINGS = {
    "no_filter": [],
    "numbers_only": ["number"],
    "medications_only": ["medication"],
    "numbers_and_medications": ["number", "medication"],
}


def scenario_settings(scenario_name):
    return {**BASELINE_SCENARIO, **DATA_SCENARIOS[scenario_name]}
