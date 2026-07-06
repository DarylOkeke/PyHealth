"""Cross-hospital covariate-shift split for eICU (Addition 2).

split_strategy="by_hospital" -- leave-hospitals-out: test = samples from the held-out
hospitals; train/val/cal = the remaining hospitals, split by patient. The NUMBER and
SELECTION of held-out hospitals is a caller parameter (`holdout_hospitals`, an explicit
set) -- the hold-out convention is locked by the team, not designed here.

Because some eICU hospitals are tiny, the framework's existing per-class calibration guard
(min class count vs ceil(1/alpha)) applies unchanged to the resulting `cal` split, so a
degenerate per-class threshold is flagged/skipped rather than produced.

eICU-specific helpers (hospital_of_stay, split_by_hospital's sample access) touch the
loaded dataset and are verified on the cluster; _hospital_partition is pure and unit-tested.
"""

from __future__ import annotations

import numpy as np
from torch.utils.data import Subset


def _hospital_partition(hosp_ids, patient_ids, holdout_hospitals, seed, ratios):
    """Pure index partition. Held-out hospitals -> test; the rest are split BY PATIENT into
    train/val/cal using the first three ratios (renormalized). Returns 4 index lists.
    Patients never cross train/val/cal; a patient's samples all share a hospital in eICU,
    so source patients never leak into the held-out test hospitals."""
    holdout = {str(h) for h in holdout_hospitals}
    test = [i for i, h in enumerate(hosp_ids) if str(h) in holdout]
    source = [i for i, h in enumerate(hosp_ids) if str(h) not in holdout]

    by_patient = {}
    for i in source:
        by_patient.setdefault(str(patient_ids[i]), []).append(i)
    patients = sorted(by_patient)
    np.random.default_rng(seed).shuffle(patients)

    r = np.array(ratios[:3], dtype=float)
    r = r / r.sum()
    n = len(patients)
    n_tr, n_va = int(r[0] * n), int(r[1] * n)
    split = {"train": patients[:n_tr],
             "val": patients[n_tr:n_tr + n_va],
             "cal": patients[n_tr + n_va:]}
    idx = {k: [i for p in ps for i in by_patient[p]] for k, ps in split.items()}
    return idx["train"], idx["val"], idx["cal"], test


def hospital_of_stay(base_dataset):
    """{patientunitstayid -> hospitalid} from the eICU patient table."""
    mapping = {}
    for patient in base_dataset.iter_patients():
        for stay in patient.get_events(event_type="patient"):
            sid = str(getattr(stay, "patientunitstayid", ""))
            hid = getattr(stay, "hospitalid", None)
            if sid and hid is not None:
                mapping[sid] = str(hid)
    return mapping


def split_by_hospital(samples, hosp_of_stay, holdout_hospitals, seed, ratios):
    """Leave-hospitals-out split of a SampleDataset. `hosp_of_stay` maps each sample's
    visit_id (patientunitstayid) to its hospitalid (see hospital_of_stay)."""
    hosp = [hosp_of_stay.get(str(samples[i]["visit_id"])) for i in range(len(samples))]
    pat = [str(samples[i]["patient_id"]) for i in range(len(samples))]
    tr, va, ca, te = _hospital_partition(hosp, pat, holdout_hospitals, seed, ratios)
    return (Subset(samples, tr), Subset(samples, va),
            Subset(samples, ca), Subset(samples, te))
