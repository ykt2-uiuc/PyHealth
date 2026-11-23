pyhealth.tasks.PatientSummarizationMIMIC4
=========================================

Patient summarization on MIMIC-IV-Note discharge notes, following the dataset
construction of Hegselmann et al., `A Data-Centric Approach To Generate
Faithful and High Quality Patient Summaries with Large Language Models
<https://arxiv.org/abs/2402.15422>`_ (CHIL 2024). The Brief Hospital Course is
the input and the Discharge Instructions section is the target summary. It runs
on :class:`~pyhealth.datasets.MIMIC4NoteDataset`.

.. code-block:: python

    from pyhealth.datasets import MIMIC4NoteDataset
    from pyhealth.tasks import PatientSummarizationMIMIC4

    dataset = MIMIC4NoteDataset(
        root="/path/to/mimic-iv-note/2.2", tables=["discharge"]
    )
    samples = dataset.set_task(PatientSummarizationMIMIC4(max_context_chars=4000))

.. autoclass:: pyhealth.tasks.patient_summarization_mimic4.PatientSummarizationMIMIC4
    :members:
    :undoc-members:
    :show-inheritance:

Helper functions
----------------

.. autofunction:: pyhealth.tasks.patient_summarization_mimic4.split_sections

.. autofunction:: pyhealth.tasks.patient_summarization_mimic4.get_section

.. autofunction:: pyhealth.tasks.patient_summarization_mimic4.count_sentences

.. autofunction:: pyhealth.tasks.patient_summarization_mimic4.deid_per_word
