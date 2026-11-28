from vocabulary import DAYS, DIAGNOSIS_DRUG_ROWS, DRUGS

ADMISSION_PHRASINGS = [
    "Patient was admitted with {diagnosis}.",
    "Patient presented to the hospital and was found to have {diagnosis}.",
    "Admitted for management of {diagnosis}.",
]

TREATMENT_PHRASINGS = [
    "Treated with {treatment_drug} {dose} mg daily for {days} days with improvement.",
    "Started on {treatment_drug} {dose} mg and completed a {days} day course.",
    "Received {treatment_drug} {dose} mg for {days} days and symptoms resolved.",
]

HOME_DRUG_PHRASINGS = [
    "Home {home_drug} was continued.",
    "Continued on home {home_drug}.",
]

STOPPED_DRUG_PHRASINGS = [
    "Home {stopped_drug} was stopped during the admission.",
    "{stopped_drug} was held and discontinued at discharge.",
]

SUMMARY_DIAGNOSIS_PHRASINGS = [
    "You were in the hospital for {diagnosis}.",
    "You came to the hospital because of {diagnosis}.",
]

SUMMARY_TREATMENT_PHRASINGS = [
    "You were treated with {treatment_drug} {dose} mg for {days} days.",
    "We gave you {treatment_drug} {dose} mg for {days} days.",
]

SUMMARY_HOME_DRUG_PHRASINGS = [
    "Please keep taking your {home_drug}.",
    "Continue your {home_drug} at home.",
]

SUMMARY_STOPPED_DRUG_PHRASINGS = [
    "Do not take your {stopped_drug} until you see your doctor.",
    "We stopped your {stopped_drug}.",
]

SUMMARY_CLOSING_PHRASINGS = [
    "Follow up with your primary care doctor.",
    "Please see your doctor at your next appointment.",
]


def create_clean_pair(random_generator):
    diagnosis_drug_row = random_generator.choice(DIAGNOSIS_DRUG_ROWS)
    diagnosis = diagnosis_drug_row["diagnosis"]
    treatment_drug = diagnosis_drug_row["treatment_drug"]
    dose = diagnosis_drug_row["dose_mg"]
    other_drugs = [drug for drug in DRUGS if drug != treatment_drug]
    home_drug, stopped_drug = random_generator.sample(other_drugs, 2)
    days = random_generator.choice([day for day in DAYS if str(day) not in dose])

    slot_values = {
        "diagnosis": diagnosis,
        "treatment_drug": treatment_drug,
        "home_drug": home_drug,
        "stopped_drug": stopped_drug,
        "dose": dose,
        "days": days,
    }

    context_sentences = [
        random_generator.choice(ADMISSION_PHRASINGS),
        random_generator.choice(TREATMENT_PHRASINGS),
        random_generator.choice(HOME_DRUG_PHRASINGS),
        random_generator.choice(STOPPED_DRUG_PHRASINGS),
        "Patient was stable at discharge.",
    ]
    summary_sentences = [
        random_generator.choice(SUMMARY_DIAGNOSIS_PHRASINGS),
        random_generator.choice(SUMMARY_TREATMENT_PHRASINGS),
        random_generator.choice(SUMMARY_HOME_DRUG_PHRASINGS),
        random_generator.choice(SUMMARY_STOPPED_DRUG_PHRASINGS),
        random_generator.choice(SUMMARY_CLOSING_PHRASINGS),
    ]

    context = " ".join(context_sentences).format(**slot_values)
    summary = " ".join(summary_sentences).format(**slot_values)
    context = context[0].upper() + context[1:]

    return {
        "context": context,
        "summary": summary,
        "diagnosis": diagnosis,
        "treatment_drug": treatment_drug,
        "dose": dose,
        "stopped_drug": stopped_drug,
    }


def create_clean_pairs(count, random_generator):
    return [create_clean_pair(random_generator) for _ in range(count)]
