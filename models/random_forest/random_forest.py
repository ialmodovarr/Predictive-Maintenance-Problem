#!/usr/bin/env python3
"""Binary HEALTHY/FAILED Random Forest for paired printer API + sensor features.

Run from the project root:
  python random_forest.py setup
  # Edit models/random_forest/training_pairs.csv to list manually selected pairs
  python random_forest.py train
  python random_forest.py predict --api FAILED_printer_features.csv --sensor FAILED_sensor_features.csv

Expected folders:
  data/api_preprocessed/
  data/sensor_preprocessed/
  models/random_forest/artifacts/

Both filenames must start with HEALTHY_ or FAILED_.
Sensor/API rows are aligned by relative recording progress as a baseline.
This assumes each pair covers approximately the same complete run; it is NOT
verified timestamp synchronization. Add a shared clock/time origin before
relying on precise event-level alignment.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent
API_DIR = ROOT / "data" / "api_preprocessed"
SENSOR_DIR = ROOT / "data" / "sensor_preprocessed"
MODEL_DIR = ROOT / "models" / "random_forest"
ARTIFACTS = MODEL_DIR / "artifacts"
MANIFEST = MODEL_DIR / "training_pairs.csv"
MODEL = ARTIFACTS / "random_forest_pipeline.joblib"
METADATA = ARTIFACTS / "model_metadata.json"

LABELS = {"HEALTHY": 0, "FAILED": 1}
META_COLUMNS = {
    "source_file", "laptop_time", "timestamp", "datetime", "date",
    "section", "window_start_s", "window_end_s", "sample_index_start",
    "sample_index_end", "board_time_start_us", "board_time_end_us",
    "elapsed_seconds", "run_id", "label", "target", "failure",
}


def ensure_dirs():
    for folder in (API_DIR, SENSOR_DIR, ARTIFACTS):
        folder.mkdir(parents=True, exist_ok=True)


def label_from_filename(path):
    name = Path(path).name.upper()
    for label, value in LABELS.items():
        if name.startswith(label + "_") or name == label:
            return value
    raise ValueError(
        f"{Path(path).name}: filename must begin with HEALTHY_ or FAILED_."
    )


def resolve_file(value, default_dir):
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    for candidate in (ROOT / p, default_dir / p):
        if candidate.exists():
            return candidate
    return default_dir / p


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"No rows in {path}")
    return df


def numeric_features(df):
    df = df.drop(
        columns=[c for c in df.columns if c.lower() in META_COLUMNS],
        errors="ignore",
    ).copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.select_dtypes(include=[np.number])
    return df.dropna(axis=1, how="all")


def align_pair(api_df, sensor_df):
    """Interpolate sensor features onto API row positions (relative progress)."""
    api = numeric_features(api_df).reset_index(drop=True)
    sensor = numeric_features(sensor_df).reset_index(drop=True)
    if api.empty or sensor.empty:
        raise ValueError("One file has no usable numeric feature columns.")

    api.columns = ["api_" + str(c) for c in api.columns]
    x_api = np.linspace(0, 1, len(api))
    x_sensor = np.linspace(0, 1, len(sensor))
    aligned = {}
    for col in sensor.columns:
        vals = sensor[col].to_numpy(dtype=float)
        valid = np.isfinite(vals)
        if valid.sum() == 0:
            aligned["sensor_" + str(col)] = np.full(len(api), np.nan)
        elif valid.sum() == 1:
            aligned["sensor_" + str(col)] = np.full(len(api), vals[valid][0])
        else:
            aligned["sensor_" + str(col)] = np.interp(
                x_api, x_sensor[valid], vals[valid]
            )
    return pd.concat([api, pd.DataFrame(aligned)], axis=1).replace(
        [np.inf, -np.inf], np.nan
    )


def load_pair(api_path, sensor_path):
    ya, ys = label_from_filename(api_path), label_from_filename(sensor_path)
    if ya != ys:
        raise ValueError(
            f"Pair labels disagree: {api_path.name} and {sensor_path.name}"
        )
    return align_pair(read_csv(api_path), read_csv(sensor_path)), ya


def setup(_):
    ensure_dirs()
    if not MANIFEST.exists():
        pd.DataFrame(columns=["api_file", "sensor_file", "notes"]).to_csv(
            MANIFEST, index=False
        )
    print("Folders ready.")
    print(f"Add manually selected matching file pairs to:\n  {MANIFEST}")
    print("\nExample rows:")
    print("api_file,sensor_file,notes")
    print("HEALTHY_printer_features_run1.csv,HEALTHY_sensor_features_run1.csv,healthy run")
    print("FAILED_printer_features_run2.csv,FAILED_sensor_features_run2.csv,failed run")


def read_manifest():
    if not MANIFEST.exists():
        raise FileNotFoundError("Run `python random_forest.py setup` first.")
    df = pd.read_csv(MANIFEST)
    if not {"api_file", "sensor_file"}.issubset(df.columns):
        raise ValueError("Manifest needs api_file and sensor_file columns.")
    return df.dropna(subset=["api_file", "sensor_file"])


def train(args):
    ensure_dirs()
    manifest = read_manifest()
    if len(manifest) < 2:
        raise ValueError("Add at least two paired runs to training_pairs.csv.")

    frames, labels, groups, run_names = [], [], [], []
    for _, row in manifest.iterrows():
        ap = resolve_file(str(row.api_file), API_DIR)
        sp = resolve_file(str(row.sensor_file), SENSOR_DIR)
        X_run, y = load_pair(ap, sp)
        run_id = ap.stem + "__" + sp.stem
        frames.append(X_run)
        labels.extend([y] * len(X_run))
        groups.extend([run_id] * len(X_run))
        run_names.append(run_id)
        print(f"Loaded {run_id}: {len(X_run)} windows, "
              f"{'FAILED' if y else 'HEALTHY'}")

    X = pd.concat(frames, ignore_index=True, sort=False)
    y = np.asarray(labels)
    groups = np.asarray(groups)
    if len(np.unique(y)) != 2:
        raise ValueError("Training data must include both HEALTHY and FAILED runs.")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("rf", RandomForestClassifier(
            n_estimators=args.trees,
            min_samples_leaf=args.min_leaf,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Hold out whole runs, never random rows from the same run.
    split = None
    if len(np.unique(groups)) >= 4:
        splitter = GroupShuffleSplit(
            n_splits=50, test_size=args.test_size, random_state=42
        )
        for tr, te in splitter.split(X, y, groups):
            if len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2:
                split = (tr, te)
                break

    if split:
        tr, te = split
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        prob = pipe.predict_proba(X.iloc[te])[:, list(pipe.classes_).index(1)]
        print("\nRun-held-out evaluation:")
        print(f"Accuracy:  {accuracy_score(y[te], pred):.3f}")
        print(f"Precision: {precision_score(y[te], pred, zero_division=0):.3f}")
        print(f"Recall:    {recall_score(y[te], pred, zero_division=0):.3f}")
        print(f"F1:        {f1_score(y[te], pred, zero_division=0):.3f}")
        print(f"ROC AUC:   {roc_auc_score(y[te], prob):.3f}")
        print("Confusion matrix [true rows, predicted columns; 0=HEALTHY, 1=FAILED]:")
        print(confusion_matrix(y[te], pred))
        print(classification_report(
            y[te], pred, target_names=["HEALTHY", "FAILED"], zero_division=0
        ))
    else:
        print("\nNot enough independent runs for a valid run-held-out split "
              "containing both classes. No training-set score will be reported.")

    # Final model is fit on every manually selected run.
    pipe.fit(X, y)
    joblib.dump(pipe, MODEL)
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "classes": {"0": "HEALTHY", "1": "FAILED"},
        "training_runs": run_names,
        "number_of_windows": int(len(X)),
        "number_of_runs": len(run_names),
        "feature_columns": list(X.columns),
        "alignment": "relative recording progress; validate shared timing before deployment",
        "probability_note": "predict_proba FAILED score; not necessarily calibrated real-world failure probability",
    }
    METADATA.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": pipe.named_steps["rf"].feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(ARTIFACTS / "feature_importance.csv", index=False)
    manifest.to_csv(ARTIFACTS / "training_pairs_snapshot.csv", index=False)
    print(f"\nSaved model: {MODEL}")
    print(f"Saved artifacts in: {ARTIFACTS}")


def predict(args):
    if not MODEL.exists() or not METADATA.exists():
        raise FileNotFoundError("Train the model first: python random_forest.py train")
    ap = resolve_file(args.api, API_DIR)
    sp = resolve_file(args.sensor, SENSOR_DIR)
    X, _ = load_pair(ap, sp)
    meta = json.loads(METADATA.read_text(encoding="utf-8"))
    X = X.reindex(columns=meta["feature_columns"])
    pipe = joblib.load(MODEL)
    probs = pipe.predict_proba(X)
    failed_index = list(pipe.classes_).index(1)
    failed_score = probs[:, failed_index]
    pred = failed_score >= args.threshold

    output = pd.DataFrame({
        "window": np.arange(len(X)),
        "predicted_label": np.where(pred, "FAILED", "HEALTHY"),
        "failure_likelihood_percent": np.round(failed_score * 100, 2),
    })
    out = Path(args.output).expanduser()
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out, index=False)
    print(f"Predictions saved: {out}")
    print(f"Mean FAILED score: {failed_score.mean() * 100:.2f}%")
    print(f"Median FAILED score: {np.median(failed_score) * 100:.2f}%")
    print(f"Windows predicted FAILED: {pred.mean() * 100:.2f}%")
    print("The percentage is the model's FAILED-class score, not automatically "
          "a calibrated probability of future failure.")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("setup")
    p.set_defaults(func=setup)
    p = sub.add_parser("train")
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--min-leaf", type=int, default=2)
    p.add_argument("--test-size", type=float, default=0.25)
    p.set_defaults(func=train)
    p = sub.add_parser("predict")
    p.add_argument("--api", required=True)
    p.add_argument("--sensor", required=True)
    p.add_argument("--threshold", type=float, default=0.50)
    p.add_argument("--output", default="models/random_forest/predictions/latest_prediction.csv")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
