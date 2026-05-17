#!/usr/bin/env python3
"""Generate clean reviewer-facing release tables from formal prediction assets.

This script performs no LLM/API calls. It only reads existing formal prediction
CSVs, removes process-only fields, normalizes task-specific labels into the
manuscript event-positive convention, and writes compact public-release CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT.parent / "magman-llm-pub"
OUT_DIR = ROOT / "public_release"


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    system: str
    model: str
    provider_channel: str
    prompt_variant: str
    task_type: str
    task_label1_meaning: str
    event_positive_class: str
    invert_for_event: bool
    evidence_role: str
    source_csv: Path


RUNS = [
    RunSpec(
        "fe3o4_preview_zero_shot",
        "Fe3O4",
        "deepseek-v3.2-exp",
        "historical-preview",
        "zero-shot",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 Fe3O4 zero-shot historical anchor",
        PUB / "original_dataset/02_main_data/deepseek-v3.2-exp-default.csv",
    ),
    RunSpec(
        "fe3o4_preview_fs1",
        "Fe3O4",
        "deepseek-v3.2-exp",
        "historical-preview",
        "FS-1",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 Fe3O4 FS-1 historical anchor",
        PUB / "original_dataset/02_main_data/deepseek-v3.2-exp_v3.csv",
    ),
    RunSpec(
        "fe3o4_preview_fs2",
        "Fe3O4",
        "deepseek-v3.2-exp",
        "historical-preview",
        "FS-2",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 and Fig. 8 Fe3O4 FS-2 historical anchor",
        PUB / "original_dataset/02_main_data/deepseek-v3.2-exp_v10.csv",
    ),
    RunSpec(
        "fe3o4_stable_fs2_run1",
        "Fe3O4",
        "deepseek-v3.2",
        "cherryin",
        "FS-2",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 caption stable-release continuity check",
        ROOT
        / "Fe3O4_t5_reproducibility/04_predictions/deepseek_v32agent_formal/fe3o4_deepseek_v32agent_run1_predictions.csv",
    ),
    RunSpec(
        "fe3o4_stable_fs2_run2",
        "Fe3O4",
        "deepseek-v3.2",
        "cherryin",
        "FS-2",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 caption stable-release continuity check",
        ROOT
        / "Fe3O4_t5_reproducibility/04_predictions/deepseek_v32agent_formal/fe3o4_deepseek_v32agent_run2_predictions.csv",
    ),
    RunSpec(
        "fe3o4_stable_fs2_run3",
        "Fe3O4",
        "deepseek-v3.2",
        "cherryin",
        "FS-2",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 caption stable-release continuity check",
        ROOT
        / "Fe3O4_t5_reproducibility/04_predictions/deepseek_v32agent_formal/fe3o4_deepseek_v32agent_run3_predictions.csv",
    ),
    RunSpec(
        "gamma_fe2o3_zero_shot_run2",
        "gamma-Fe2O3",
        "deepseek-v3.2",
        "cherryin",
        "zero-shot",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2 gamma-Fe2O3 zero-shot formal result",
        ROOT
        / "Fe2O3ga_validation/04_predictions/deepseek_v32agent_formal/fe2o3ga_deepseek_v32agent_zero_shot_run2_predictions.csv",
    ),
    RunSpec(
        "gamma_fe2o3_fs2",
        "gamma-Fe2O3",
        "deepseek-v3.2",
        "cherryin",
        "FS-2",
        "200-step convergence",
        "convergent within 200 electronic SCF steps",
        "non-convergent within 200 electronic SCF steps",
        True,
        "Table 2, Fig. 8, and SI-B gamma-Fe2O3 FS-2 formal result",
        ROOT
        / "Fe2O3ga_validation/04_predictions/deepseek_v32agent_formal/fe2o3ga_deepseek_v32agent_run1_predictions.csv",
    ),
    RunSpec(
        "mn2o3_deepseek_zero_shot",
        "Mn2O3",
        "deepseek-v3.2",
        "cherryin",
        "zero-shot",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Table 2 Mn2O3 zero-shot formal result",
        ROOT
        / "Mn2O3ga_validation/04_predictions/deepseek_v32agent_formal/mn2o3ga_deepseek_v32agent_zero_shot_run1_predictions.csv",
    ),
    RunSpec(
        "mn2o3_deepseek_fs2",
        "Mn2O3",
        "deepseek-v3.2",
        "cherryin",
        "FS-2",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Table 2, Table 3, Fig. 8, and SI-B Mn2O3 FS-2 formal result",
        ROOT
        / "Mn2O3ga_validation/04_predictions/deepseek_v32agent_formal/mn2o3ga_deepseek_v32agent_fs2_run1_predictions.csv",
    ),
    RunSpec(
        "mn2o3_deepseek_system_adapted",
        "Mn2O3",
        "deepseek-v3.2",
        "cherryin",
        "System-adapted prompt",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Table 3 Mn2O3 system-adapted formal result",
        ROOT
        / "Mn2O3ga_validation/04_predictions/deepseek_v32agent_formal/mn2o3ga_deepseek_v32agent_minus_only_run1_predictions.csv",
    ),
    RunSpec(
        "mn2o3_deepseek_indicator_augmented",
        "Mn2O3",
        "deepseek-v3.2",
        "cherryin",
        "Indicator-augmented prompt",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Table 2, Table 3, Fig. 8, and SI-B Mn2O3 adapted formal result",
        ROOT
        / "Mn2O3ga_validation/04_predictions/deepseek_v32agent_formal/mn2o3ga_deepseek_v32agent_minus_plus_run1_predictions.csv",
    ),
    RunSpec(
        "mn2o3_glm5_zero_shot",
        "Mn2O3",
        "glm-5",
        "cherryin",
        "zero-shot",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Response-letter cross-vendor supporting zero-shot result",
        ROOT / "Mn2O3ga_validation/04_predictions/formal/glm5_zero_shot_screening50_magman.csv",
    ),
    RunSpec(
        "mn2o3_glm5_fs2",
        "Mn2O3",
        "glm-5",
        "cherryin",
        "FS-2",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Response-letter cross-vendor supporting FS-2 result",
        ROOT / "Mn2O3ga_validation/04_predictions/formal/glm5_fs2_screening50_magman.csv",
    ),
    RunSpec(
        "mn2o3_v4flash_fs2",
        "Mn2O3",
        "deepseek-v4-flash",
        "cherryin",
        "FS-2",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Response-letter cross-generation supporting FS-2 result",
        ROOT
        / "Mn2O3ga_validation/04_predictions/deepseek_v32agent_formal/mn2o3ga_deepseek_v4flash_fs2_run1_predictions.csv",
    ),
    RunSpec(
        "mn2o3_v4flash_indicator_augmented",
        "Mn2O3",
        "deepseek-v4-flash",
        "cherryin",
        "Indicator-augmented prompt",
        "100-step reach",
        "reaching 100 electronic SCF steps",
        "reaching 100 electronic SCF steps",
        False,
        "Response-letter cross-generation supporting adapted result",
        ROOT
        / "Mn2O3ga_validation/04_predictions/deepseek_v32agent_formal/mn2o3ga_deepseek_v4flash_minus_plus_run1_predictions.csv",
    ),
]


def public_sample_fields(filepath: str, system: str) -> tuple[str, str, str]:
    parts = Path(str(filepath)).parts
    sample_id = Path(str(filepath)).parent.name
    if system == "Mn2O3":
        if "mn2o3optgaga" in parts:
            source_group = "optgaga_jobs"
        elif "mn2o3optga" in parts:
            source_group = "optga_jobs"
        else:
            source_group = "jobs"
    elif system == "gamma-Fe2O3":
        source_group = "gamma_fe2o3_jobs"
    else:
        source_group = "fe3o4_screening50"
    public_uid = f"{source_group}:{sample_id}"
    return sample_id, source_group, public_uid


def task_label1_score(prediction: pd.Series, confidence_score: pd.Series) -> pd.Series:
    """Map ordinal confidence to a monotonic score for task-specific label=1.

    This follows the convention used in scripts/make_fig8_roc.py: confidence 0
    maps to 0.05 and confidence 9 maps to 0.95 for the predicted class.
    """
    base = 0.05 + (confidence_score / 9.0) * 0.9
    return base.where(prediction == 1, 1.0 - base)


def roc_auc(y_true: pd.Series, score: pd.Series) -> float:
    data = pd.DataFrame({"y": y_true.astype(int), "score": score.astype(float)})
    data = data.dropna().sort_values("score", ascending=False)
    positives = int(data["y"].sum())
    negatives = int(len(data) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("ROC-AUC requires both positive and negative samples")

    fpr = [0.0]
    tpr = [0.0]
    tp = 0
    fp = 0
    for _, group in data.groupby("score", sort=False):
        tp += int(group["y"].sum())
        fp += int(len(group) - group["y"].sum())
        fpr.append(fp / negatives)
        tpr.append(tp / positives)

    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
    return auc


def binary_metrics(y_true: pd.Series, y_pred: pd.Series, score: pd.Series) -> dict[str, float | int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_samples": int(len(y_true)),
        "positive_samples": int(y_true.sum()),
        "negative_samples": int(len(y_true) - y_true.sum()),
        "accuracy": float((y_true == y_pred).mean()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(roc_auc(y_true, score)),
        "true_positives": tp,
        "false_positives": fp,
        "true_negatives": tn,
        "false_negatives": fn,
    }


def normalize_run(spec: RunSpec) -> pd.DataFrame:
    if not spec.source_csv.exists():
        raise FileNotFoundError(spec.source_csv)
    df = pd.read_csv(spec.source_csv)
    df = df[df["success"].astype(str).str.lower().eq("true")].copy()
    df["task_true_label"] = pd.to_numeric(df["true_label"], errors="raise").astype(int)
    df["task_prediction"] = pd.to_numeric(df["prediction"], errors="raise").astype(int)
    df["confidence_score"] = pd.to_numeric(df["confidence_score"], errors="raise").astype(int)
    df["n_input_steps"] = pd.to_numeric(df["n_steps"], errors="raise").astype(int)
    df["task_label1_score"] = task_label1_score(df["task_prediction"], df["confidence_score"])
    if spec.invert_for_event:
        df["event_true"] = 1 - df["task_true_label"]
        df["event_pred"] = 1 - df["task_prediction"]
        df["event_score"] = 1.0 - df["task_label1_score"]
    else:
        df["event_true"] = df["task_true_label"]
        df["event_pred"] = df["task_prediction"]
        df["event_score"] = df["task_label1_score"]

    sample_fields = df["filepath"].map(lambda value: public_sample_fields(value, spec.system))
    df["sample_id"] = sample_fields.map(lambda item: item[0])
    df["source_group"] = sample_fields.map(lambda item: item[1])
    df["public_sample_uid"] = sample_fields.map(lambda item: item[2])

    df["run_id"] = spec.run_id
    df["system"] = spec.system
    df["model"] = spec.model
    df["provider_channel"] = spec.provider_channel
    df["prompt_variant"] = spec.prompt_variant
    df["task_type"] = spec.task_type
    df["task_label1_meaning"] = spec.task_label1_meaning
    df["event_positive_class"] = spec.event_positive_class
    df["invert_for_event"] = spec.invert_for_event
    df["evidence_role"] = spec.evidence_role

    cols = [
        "run_id",
        "system",
        "model",
        "provider_channel",
        "prompt_variant",
        "evidence_role",
        "task_type",
        "task_label1_meaning",
        "event_positive_class",
        "invert_for_event",
        "sample_id",
        "source_group",
        "public_sample_uid",
        "n_input_steps",
        "task_true_label",
        "task_prediction",
        "event_true",
        "event_pred",
        "confidence_score",
        "task_label1_score",
        "event_score",
    ]
    return df[cols].sort_values(["run_id", "source_group", "sample_id"]).reset_index(drop=True)


def build_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = [
        "run_id",
        "system",
        "model",
        "provider_channel",
        "prompt_variant",
        "evidence_role",
        "task_type",
        "task_label1_meaning",
        "event_positive_class",
        "invert_for_event",
    ]
    for values, group in predictions.groupby(group_cols, sort=False):
        row = dict(zip(group_cols, values, strict=True))
        event = binary_metrics(group["event_true"], group["event_pred"], group["event_score"])
        task = binary_metrics(
            group["task_true_label"], group["task_prediction"], group["task_label1_score"]
        )
        row.update({f"event_{key}": value for key, value in event.items()})
        row.update(
            {
                "task_accuracy": task["accuracy"],
                "task_precision": task["precision"],
                "task_recall": task["recall"],
                "task_f1": task["f1"],
                "task_auc": task["auc"],
                "task_true_positives": task["true_positives"],
                "task_false_positives": task["false_positives"],
                "task_true_negatives": task["true_negatives"],
                "task_false_negatives": task["false_negatives"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_table2(metrics: pd.DataFrame) -> pd.DataFrame:
    def auc(run_id: str) -> float:
        return float(metrics.loc[metrics["run_id"] == run_id, "event_auc"].iloc[0])

    return pd.DataFrame(
        [
            {
                "system": "Fe3O4",
                "task_type": "200-step convergence",
                "zero_shot_auc": auc("fe3o4_preview_zero_shot"),
                "fs1_auc": auc("fe3o4_preview_fs1"),
                "fs2_auc": auc("fe3o4_preview_fs2"),
                "adapted_auc": pd.NA,
                "delta_zero_shot_to_fs2": auc("fe3o4_preview_fs2")
                - auc("fe3o4_preview_zero_shot"),
                "source_runs": "fe3o4_preview_zero_shot;fe3o4_preview_fs1;fe3o4_preview_fs2",
            },
            {
                "system": "gamma-Fe2O3",
                "task_type": "200-step convergence",
                "zero_shot_auc": auc("gamma_fe2o3_zero_shot_run2"),
                "fs1_auc": pd.NA,
                "fs2_auc": auc("gamma_fe2o3_fs2"),
                "adapted_auc": pd.NA,
                "delta_zero_shot_to_fs2": auc("gamma_fe2o3_fs2")
                - auc("gamma_fe2o3_zero_shot_run2"),
                "source_runs": "gamma_fe2o3_zero_shot_run2;gamma_fe2o3_fs2",
            },
            {
                "system": "Mn2O3",
                "task_type": "100-step reach",
                "zero_shot_auc": auc("mn2o3_deepseek_zero_shot"),
                "fs1_auc": pd.NA,
                "fs2_auc": auc("mn2o3_deepseek_fs2"),
                "adapted_auc": auc("mn2o3_deepseek_indicator_augmented"),
                "delta_zero_shot_to_fs2": auc("mn2o3_deepseek_fs2")
                - auc("mn2o3_deepseek_zero_shot"),
                "source_runs": "mn2o3_deepseek_zero_shot;mn2o3_deepseek_fs2;mn2o3_deepseek_indicator_augmented",
            },
        ]
    )


def build_table3(metrics: pd.DataFrame) -> pd.DataFrame:
    run_order = [
        "mn2o3_deepseek_fs2",
        "mn2o3_deepseek_system_adapted",
        "mn2o3_deepseek_indicator_augmented",
    ]
    rows = []
    for run_id in run_order:
        row = metrics.loc[metrics["run_id"] == run_id].iloc[0]
        rows.append(
            {
                "prompt_variant": row["prompt_variant"],
                "auc": row["event_auc"],
                "accuracy": row["event_accuracy"],
                "false_positives": row["event_false_positives"],
                "false_negatives": row["event_false_negatives"],
                "positive_class": row["event_positive_class"],
                "source_run": run_id,
            }
        )
    return pd.DataFrame(rows)


def write_readme() -> None:
    readme = """# Public release normalized data

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
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions = pd.concat([normalize_run(spec) for spec in RUNS], ignore_index=True)
    metrics = build_metrics(predictions)
    table2 = build_table2(metrics)
    table3 = build_table3(metrics)

    predictions.to_csv(OUT_DIR / "normalized_predictions.csv", index=False)
    metrics.to_csv(OUT_DIR / "normalized_metrics.csv", index=False)
    table2.to_csv(OUT_DIR / "table2_cross_system_auc.csv", index=False)
    table3.to_csv(OUT_DIR / "table3_mn2o3_prompt_adaptation.csv", index=False)
    write_readme()
    print(f"wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
