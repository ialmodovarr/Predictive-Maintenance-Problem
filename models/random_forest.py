"""
Multi-sensor baseline Random Forest for Ultimaker S3 predictive maintenance.

Designed for a future merged dataset containing synchronized API telemetry,
accelerometer, temperature-sensor, and filament-motion measurements.

Recommended input: one CSV with one row per synchronized sample and columns like:
    timestamp, run_id, state,
    accel_x, accel_y, accel_z,
    hotend_temp_c, external_temp_c,
    filament_commanded_mm, filament_actual_mm

Column names are flexible: the script uses numeric columns automatically and
excludes common metadata/target columns. Rename the target column or pass
--target. The target must be a known experimental label for each sample, such
as healthy/degrading/failing.

For raw time-series input, the script summarizes numeric sensor measurements
inside consecutive windows (mean, standard deviation, min/max, range, RMS,
quantiles, and trend/slope). This creates one feature row per window.
If your CSV already contains engineered feature rows, pass --mode features.

IMPORTANT:
- This script does not synchronize separate files or infer ground-truth labels.
  Merge/resample sensor streams to a common timeline and label samples/windows
  from your controlled experiments before training.
- Keep run_id (or print_id) in the data for grouped train/test splitting.
- Do not include future-looking columns or outcomes that would reveal the label.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline


METADATA_NAMES = {
    "timestamp", "time", "datetime", "date", "laptop_time", "sample_time",
    "index", "unnamed: 0", "run_id", "print_id", "experiment_id",
    "failure_type", "notes", "state", "label", "target",
}


def get_numeric_sensor_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = METADATA_NAMES | {target.lower()}
    cols = []
    for col in df.columns:
        if col.lower().strip() in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def window_features(
    df: pd.DataFrame,
    target: str,
    window_size: int,
    run_col: str | None,
) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Summarize each consecutive window of numeric sensor samples."""
    numeric_cols = get_numeric_sensor_columns(df, target)
    if not numeric_cols:
        raise ValueError("No numeric sensor columns found. Check the input CSV.")

    if run_col is None:
        df = df.copy()
        df["_temporary_run_id"] = "single_run"
        run_col = "_temporary_run_id"

    feature_rows: list[dict] = []
    labels: list[str] = []
    groups: list[str] = []

    for run_id, run_df in df.groupby(run_col, sort=False, dropna=False):
        run_df = run_df.reset_index(drop=True)
        for start in range(0, len(run_df), window_size):
            chunk = run_df.iloc[start:start + window_size]
            if chunk.empty:
                continue

            # Use the most common known label in this window.
            known_labels = chunk[target].dropna().astype(str).str.strip().str.lower()
            known_labels = known_labels[known_labels != ""]
            if known_labels.empty:
                continue
            label = known_labels.mode().iloc[0]

            row: dict[str, float] = {}
            for col in numeric_cols:
                values = pd.to_numeric(chunk[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if values.empty:
                    row[f"{col}__mean"] = np.nan
                    row[f"{col}__std"] = np.nan
                    row[f"{col}__min"] = np.nan
                    row[f"{col}__max"] = np.nan
                    row[f"{col}__range"] = np.nan
                    row[f"{col}__rms"] = np.nan
                    row[f"{col}__q25"] = np.nan
                    row[f"{col}__q75"] = np.nan
                    row[f"{col}__slope"] = np.nan
                    continue

                arr = values.to_numpy(dtype=float)
                row[f"{col}__mean"] = float(np.mean(arr))
                row[f"{col}__std"] = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                row[f"{col}__min"] = float(np.min(arr))
                row[f"{col}__max"] = float(np.max(arr))
                row[f"{col}__range"] = float(np.max(arr) - np.min(arr))
                row[f"{col}__rms"] = float(np.sqrt(np.mean(arr ** 2)))
                row[f"{col}__q25"] = float(np.quantile(arr, 0.25))
                row[f"{col}__q75"] = float(np.quantile(arr, 0.75))
                # Approximate trend across samples within the window.
                row[f"{col}__slope"] = (
                    float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
                    if len(arr) > 1 and np.ptp(arr) > 0 else 0.0
                )

            row["window_start_row"] = int(start)
            feature_rows.append(row)
            labels.append(label)
            groups.append(str(run_id))

    if not feature_rows:
        raise ValueError("No labeled windows were created. Check target labels.")

    X = pd.DataFrame(feature_rows).drop(
        columns=["window_start_row"], errors="ignore"
    )
    y = pd.Series(labels, name=target)
    group_series = pd.Series(groups, name=run_col)
    return X, y, group_series


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a multi-sensor baseline Random Forest classifier."
    )
    parser.add_argument("--data", default="data/processed/labeled_multisensor_data.csv")
    parser.add_argument("--target", default="state")
    parser.add_argument(
        "--mode", choices=["raw", "features"], default="raw",
        help="raw: summarize consecutive sensor samples into windows; "
             "features: use each row as an already-engineered feature vector.",
    )
    parser.add_argument(
        "--window-size", type=int, default=20,
        help="Number of synchronized samples per window in raw mode (default: 20).",
    )
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="models/multisensor_random_forest")
    args = parser.parse_args()

    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {data_path}\n"
            "Create a merged, labeled CSV with synchronized sensor measurements "
            "and a target column (default: state), then rerun."
        )
    if args.window_size < 1:
        raise ValueError("--window-size must be at least 1.")

    df = pd.read_csv(data_path)
    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Available columns: "
            f"{df.columns.tolist()}"
        )

    df = df.dropna(subset=[args.target]).copy()
    df[args.target] = df[args.target].astype(str).str.strip().str.lower()
    df = df[df[args.target] != ""]
    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} labeled sample rows found. Collect more labeled "
            "data across multiple experiments before evaluating a baseline."
        )

    run_col = next((c for c in ["run_id", "print_id", "experiment_id"] if c in df.columns), None)

    if args.mode == "raw":
        X, y, groups = window_features(df, args.target, args.window_size, run_col)
    else:
        numeric_cols = get_numeric_sensor_columns(df, args.target)
        if not numeric_cols:
            raise ValueError("No numeric sensor feature columns found.")
        X = df[numeric_cols].copy()
        X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
        y = df[args.target].reset_index(drop=True)
        groups = df[run_col].astype(str).reset_index(drop=True) if run_col else None
        X = X.reset_index(drop=True)

    if X.empty:
        raise ValueError("No usable features found.")
    if y.nunique() < 2:
        raise ValueError("The target must contain at least two distinct classes.")

    # Split by complete print/experiment runs when possible.
    if groups is not None and groups.nunique() >= 2:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=args.test_size, random_state=args.seed
        )
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        split_method = f"group split by {run_col or 'run ID'}"
    else:
        warnings.warn(
            "No usable run IDs with at least two unique runs. Using a stratified "
            "random split can overestimate performance on time-series data. "
            "Add run_id to evaluate on held-out experiments.",
            RuntimeWarning,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed, stratify=y
        )
        split_method = "stratified random split (no usable run grouping)"

    if y_train.nunique() < 2:
        raise ValueError(
            "Training split has fewer than two classes. Collect more independent "
            "runs per class or adjust the split."
        )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=args.trees,
            class_weight="balanced",
            random_state=args.seed,
            n_jobs=-1,
            min_samples_leaf=2,
        )),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(
        y_test, predictions, labels=labels, output_dict=True, zero_division=0
    )
    metrics = {
        "input_file": str(data_path),
        "mode": args.mode,
        "window_size_samples": args.window_size if args.mode == "raw" else None,
        "rows_or_windows_used": int(len(X)),
        "feature_count": int(X.shape[1]),
        "feature_columns": X.columns.tolist(),
        "classes": labels,
        "train_rows_or_windows": int(len(X_train)),
        "test_rows_or_windows": int(len(X_test)),
        "split_method": split_method,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }

    joblib.dump(
        {"pipeline": model, "feature_columns": X.columns.tolist(),
         "classes": labels, "target": args.target, "mode": args.mode,
         "window_size": args.window_size if args.mode == "raw" else None},
        output_dir / "multisensor_baseline_random_forest.joblib",
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "correct": y_test.to_numpy() == predictions,
    }, index=X_test.index).to_csv(output_dir / "test_predictions.csv", index_label="window_or_row")

    forest = model.named_steps["classifier"]
    pd.DataFrame({
        "feature": X.columns,
        "importance": forest.feature_importances_,
    }).sort_values("importance", ascending=False).to_csv(
        output_dir / "feature_importance.csv", index=False
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45, values_format="d")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nMulti-sensor baseline Random Forest")
    print("----------------------------------")
    print(f"Mode: {args.mode}")
    print(f"Rows/windows used: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {labels}")
    print(f"Split: {split_method}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Macro F1: {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Artifacts saved to: {output_dir.resolve()}")
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


if __name__ == "__main__":
    main()
