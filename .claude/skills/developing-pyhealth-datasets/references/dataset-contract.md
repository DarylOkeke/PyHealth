# PyHealth Dataset Contract

## Contents

- Architecture
- Mapping decisions
- Extension choices
- Correctness invariants
- Test matrix
- Review questions

## Architecture

Trace dataset work through this chain:

```text
source files
  -> DatasetConfig / dataset-specific loader
  -> normalized event rows
  -> Patient
  -> BaseTask
  -> raw sample dictionaries
  -> SampleBuilder processors
  -> SampleDataset
  -> split / dataloader / model
```

`BaseDataset.load_table()` normally produces:

- `patient_id`: string identifier used to group all records for one patient;
- `event_type`: configured table name used by `Patient.get_events()`;
- `timestamp`: normalized event time, or missing time when the source has none;
- `<table>/<attribute>`: namespaced source attributes exposed on each event.

The task is the consumer contract. Read its event queries, attribute access, temporal comparisons, required tables, sample keys, and schemas before selecting dataset mappings.

## Mapping decisions

Write a mapping table before implementation:

| Source | PyHealth field | Consumer | Evidence |
|---|---|---|---|
| source patient column | `patient_id` | patient grouping | source schema |
| source table | `event_type` | task event query | task source |
| source time column | `timestamp` | ordering/window logic | source semantics |
| source value/code | `<table>/<attribute>` | task attribute | task source |

Resolve these explicitly:

- Whether IDs identify a patient, encounter, ICU stay, study, or image.
- Whether a timestamp marks occurrence, order, collection, result, admission, or discharge.
- Whether date-only timestamps provide enough ordering for the intended task.
- Whether records with missing IDs or times are retained, rejected, or excluded by a task.
- Whether duplicate rows are meaningful repeated events.
- Whether joins are one-to-one, many-to-one, or many-to-many, and their expected row counts.
- Whether a provided split is patient-level and authoritative.
- Whether a code is raw, normalized, versioned, or mapped to another vocabulary.

If authoritative documentation and downstream behavior do not settle a choice, present the alternatives and ask the maintainer.

## Extension choices

### Configuration-backed tabular source

Use a small `BaseDataset` subclass plus YAML when `file_path`, `patient_id`, `timestamp`, attributes, and supported joins describe the source faithfully.

If custom code reads a source table before `BaseDataset.load_table()`, resolve its path and column names from the effective configuration. A public `config_path` contract is broken when preprocessing still assumes the default filename or schema.

### Specialized or typed source

Override only the necessary loading or validation hooks for sharded files, typed Parquet, remote resources, source-provided splits, or schema requirements. Preserve the normalized event contract.

### Multiple modalities

Trace how modalities are aligned by patient, encounter, study, and time. Define the prediction cutoff first. Include only observations available by that cutoff.

### Existing dataset change

Read all subclasses, callers, tasks, examples, and tests affected by the changed method or field. Preserve public constructor compatibility unless the change intentionally alters it.

## Correctness invariants

### Identity and grouping

- Convert patient identifiers consistently across every table.
- Do not confuse encounter or study identifiers with patient identifiers.
- Preserve record identifiers needed by downstream grouping or splitting.

### Time and leakage

- Choose timestamps by clinical meaning, not merely because a column parses.
- Confirm event ordering after joins and missing-time handling.
- For prediction tasks, compare each feature time with the task's prediction cutoff and outcome time.
- Apply the same information boundary to positive and negative outcomes.

### Tables and joins

- Validate requested table names against the configuration.
- Avoid adding required tables twice.
- Do not mutate a caller-provided `tables` list.
- Test join row counts and unmatched records with a hand-checkable fixture.
- Preserve source dtypes when they carry meaning, especially timestamps and identifiers.

### Caching

- List every constructor argument that can change loaded content or interpretation.
- Include a digest of the effective configuration contents in cache identity; hashing only its path still permits stale events after an in-place edit. `BaseDataset` does not currently include configuration identity automatically.
- Confirm each such argument changes the cache location or otherwise invalidates stale data.
- Verify two differently filtered instances cannot read each other's cached rows.
- Verify changing a configuration at the same path produces a new cache.
- Treat partial cache output as invalid.

### Splits and samples

- Use patient-disjoint splits unless a documented workflow requires another unit.
- Verify subset index mappings refer to the subset's local indices.
- Check every raw sample contains the task's declared input and output keys.
- Confirm processor output shape and type with at least one real sample from the synthetic fixture.

### Packaging

- Export new public datasets from `pyhealth.datasets`.
- Add API documentation following neighboring datasets.
- Add optional dependencies only when required, and guard imports so unrelated package imports still work.
- Follow the current repository contribution rules and nearby naming conventions.

## Test matrix

Use the smallest synthetic fixture that can prove each relevant contract.

| Layer | Minimum evidence |
|---|---|
| Constructor | defaults, invalid table/configuration, caller arguments unchanged |
| Table loading | canonical columns, dtypes, timestamps, namespaced attributes |
| Joining | expected matched/unmatched rows and no accidental multiplication |
| Patient | correct event types, attributes, order, and patient grouping |
| Filtering/splits | exact patient membership and disjointness |
| Cache | content-changing options produce independent cached results |
| Task integration | intended task creates correctly keyed samples |
| SampleDataset | processors fit, one sample loads, mappings are valid |
| Public API | top-level import works with required and optional dependencies |

Avoid tests that only assert the loader runs. Assert exact hand-computable content and the downstream behavior the dataset exists to support.

## Review questions

Before approving a dataset change, answer:

1. What raw fact does each normalized field represent?
2. Which task reads it, and under what time window?
3. Can outcome status change which inputs are included?
4. Can two constructor configurations collide in the cache?
5. Can a join change the cohort or duplicate events?
6. Are patient, encounter, and study boundaries preserved?
7. Does the synthetic test reach the task and `SampleDataset`, or stop at parsing?
8. Are unsupported data, dependencies, and environments reported as untested rather than passing?
