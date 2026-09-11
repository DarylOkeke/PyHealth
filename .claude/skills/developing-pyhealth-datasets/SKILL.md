---
name: developing-pyhealth-datasets
description: Use when adding, modifying, or reviewing PyHealth dataset loaders, YAML table configurations, normalized events, dataset caching, dataset-to-task integration, or dataset tests under pyhealth/datasets.
---

# Developing PyHealth Datasets

## Goal

Translate source-specific files into PyHealth's common patient and event model without changing their clinical meaning. A loader is complete only when its events work with the intended task and downstream `SampleDataset` flow.

## Workflow

1. **Map the contract before coding.** Read the current `BaseDataset`, the closest dataset with the same storage format or modality, and every intended task consumer. Record:
   `raw column -> event_type/timestamp/namespaced attribute -> task field`.
2. **Surface decisions.** Do not guess the patient or encounter identifier, timestamp, temporal boundary, join cardinality, duplicate policy, missing-value behavior, split meaning, or clinical code field. Use source documentation or ask the maintainer.
3. **Choose the smallest extension point.** Prefer a YAML-backed `BaseDataset` subclass when its loading rules are sufficient. Override loading, schema validation, or cache behavior only when the source format requires it.
4. **Keep loading and cache identity aligned with the effective configuration.** Custom prefilters and validation must use configured file paths and column mappings rather than hardcoded defaults. Any constructor option that changes loaded patients, rows, fields, or interpretation must produce an isolated cache. Hash configuration contents, not only `config_path`, so editing a YAML file in place cannot reuse stale events. Apply patient filters consistently to every loaded table.
5. **Complete the public surface.** Update the dataset class, configuration, export, API documentation, dependencies, and example only when each is required by the change.
6. **Verify the whole path with synthetic data.** Exercise raw files, normalized events, `Patient`, the intended task, `SampleDataset`, and splitting or collation when affected. Run the targeted tests, repository contribution checks, and repository formatter check.

Read [references/dataset-contract.md](references/dataset-contract.md) before designing or reviewing a dataset change.

## Required checks

- `patient_id` is stable and represented consistently across tables.
- `event_type`, timestamp, and namespaced attributes match their task consumers.
- Event timing cannot expose information occurring after the prediction outcome.
- Examples and smoke-test tasks do not invent a prediction cutoff absent from the documented task contract.
- Joins do not silently multiply or discard records beyond the documented contract.
- Caller-owned arguments are not mutated and required tables are not duplicated.
- Content-changing options and configuration contents cannot reuse stale caches.
- Subsets and splits remain patient-safe and internally indexable.
- Optional dependencies fail clearly when absent and do not break unrelated imports.
- Tests use small synthetic fixtures, fixed seeds where randomness matters, and no restricted clinical data.

## Review standard

Trace the execution path before reporting a defect. Separate unsupported assumptions, environment failures, and design preferences from reproducible correctness problems.
