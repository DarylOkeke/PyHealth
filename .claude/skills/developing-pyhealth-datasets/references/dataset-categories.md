# PyHealth Dataset Categories

Starter map for the CPBench "one per category" plan: pick one dataset per data
modality so the benchmark covers each kind of input the tasks and CP methods
must handle. Categories follow the agreed grouping: codes, text, signal,
multimodal.

## Codes (structured clinical codes)

The bulk of PyHealth. Datasets: `pyhealth/datasets/mimic3.py`,
`mimic4.py`, `eicu.py`, `omop.py`, `meds.py`, `ehrshot.py`, `support2.py`,
`bmd_hs.py`, `cardiology.py`, `mimicextract.py`, `dreamt.py`.

Starter: MIMIC-IV (`mimic4.py`). Widest task coverage in the repo
(mortality, readmission, length of stay), and the existing conformal
prediction examples (`feat/mortality-conformal-ehr-example`,
`feat/readmission-conformal-ehr-example`, `feat/los-conformal-ehr-example`)
are built on it, so new CP methods can be compared against a known baseline.

## Text (clinical notes)

Datasets: `pyhealth/datasets/medical_transcriptions.py`
(transcriptions scraped from mtsamples.com), `physionet_deid.py`
(de-identification, also text-oriented).

Starter: `MedicalTranscriptionsDataset`. Small, public, and lightweight,
so it is the cheapest way to exercise the text path end to end without
gated data access.

## Signal (biosignals)

Datasets: `pyhealth/datasets/eegbci.py` (PhysioNet EEG motor
movement/imagery), `tuab.py`, `tuev.py` (TUH EEG corpora), `shhs.py`,
`isruc.py`, `sleepedf.py` (sleep studies).

Starter: `EEGBCIDataset`. Classic public EEG benchmark with a simple
trial structure, good for validating that the signal loading and
splitting path works before touching the larger sleep corpora.

## Multimodal (codes plus another modality)

Image datasets: `pyhealth/datasets/chestxray14.py`,
`covid19_cxr.py`. The codes-plus-X-ray pairing is exercised by
`examples/mortality_prediction/multimodal_mimic4_minimal.py` (MIMIC-IV
mortality with chest X-ray input). `tcga_prad.py` covers multi-omics as
a second multimodal flavor.

Starter: the MIMIC-IV multimodal mortality setup. It reuses the codes
starter's task and cohort, so the only new variable is the added
modality, which keeps leakage review tractable (see the X-ray leakage
fix covered by `tests/core/test_multimodal_mortality_leakage.py`).

## Notes for benchmark design

- One starter per category is the minimum scope. Add a second dataset
  in a category only to test a claim the starter cannot (e.g. OMOP CDM
  versus MIMIC-IV for schema generality).
- Genomics (`clinvar.py`, `cosmic.py`) is a possible fifth category if
  the paper needs it, but it sits outside the four agreed categories.
- Every starter above must satisfy the contract in
  `references/dataset-contract.md` before it counts as benchmark-ready.
