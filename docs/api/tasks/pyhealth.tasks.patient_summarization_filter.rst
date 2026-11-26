pyhealth.tasks.patient_summarization_filter
===========================================

Rule-based filter for unsupported facts in patient summaries, for the dataset of
:doc:`pyhealth.tasks.PatientSummarizationMIMIC4`. It flags numbers and medication
names in a summary that the Brief Hospital Course does not contain, and it can
drop the sentences that hold them. ``span_scores`` compares the flags with expert
labels, using the partial-match rule of Hegselmann et al., `A Data-Centric
Approach To Generate Faithful and High Quality Patient Summaries with Large
Language Models <https://arxiv.org/abs/2402.15422>`_ (CHIL 2024).

.. code-block:: python

    from pyhealth.tasks.patient_summarization_filter import UnsupportedFactFilter

    fact_filter = UnsupportedFactFilter()
    summary = "You had chest pain. Take lisinopril 10 mg daily."
    spans = fact_filter.find(summary, "Admitted with chest pain.")
    fact_filter.drop_sentences(summary, spans)  # "You had chest pain."

.. autoclass:: pyhealth.tasks.patient_summarization_filter.UnsupportedFactFilter
    :members:
    :undoc-members:
    :show-inheritance:

Helper functions
----------------

.. autofunction:: pyhealth.tasks.patient_summarization_filter.span_scores

.. autofunction:: pyhealth.tasks.patient_summarization_filter.normalize_number
