# Public release normalized data

This directory contains clean, reviewer-facing derived tables for the COMMAT revision evidence package. The files are generated from formal prediction assets by `scripts/normalize_public_release.py` without any LLM/API calls.

## Files

- `normalized_predictions.csv`: per-sample prediction table with process fields removed.
- `normalized_metrics.csv`: per-run metrics under both task-specific label semantics and manuscript event-positive semantics.
- `table2_cross_system_auc.csv`: compact source table for the main-text cross-system prompt-tier performance table.
- `table3_mn2o3_prompt_adaptation.csv`: compact source table for the main-text Mn2O3 prompt-adaptation table.

## Label semantics

The original prediction files use task-specific `label=1` conventions. This release keeps those task labels and also adds normalized manuscript event-positive fields.

| System | Task-specific `label=1` | Manuscript event-positive class | Normalization |
|---|---|---|---|
| Fe3O4 | convergent within 200 electronic SCF steps | non-convergent within 200 electronic SCF steps | `event = 1 - task_label` |
| gamma-Fe2O3 | convergent within 200 electronic SCF steps | non-convergent within 200 electronic SCF steps | `event = 1 - task_label` |
| Mn2O3 | reaching 100 electronic SCF steps | reaching 100 electronic SCF steps | `event = task_label` |

## Removed process fields

The normalized tables intentionally exclude absolute file paths, LLM reasoning text, API timing, error fields, console logs, debug runs, legacy runs, and raw OSZICAR contents. They retain only the fields needed to audit the manuscript and response-letter numerical claims.

## Score convention

`task_label1_score` is derived from the binary prediction and ordinal confidence score using the same monotonic convention as `scripts/make_fig8_roc.py`: confidence 0 maps to 0.05 and confidence 9 maps to 0.95 for the predicted class. `event_score` applies the event-positive normalization above.
