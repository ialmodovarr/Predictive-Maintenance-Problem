#!/usr/bin/env python3
"""Random Forest component-health classifiers trained on data/runs feature files.

Run from the project root:
    python models/random_forest/random_forest.py train
    python models/random_forest/random_forest.py train --features api --exclude api_extruder_temperature_error
    python models/random_forest/random_forest.py predict --run run_0010

Data (DATA_PIPELINE.md sections 9 and 12):
- Each run's features/api_features.csv and features/sensor_features.csv are
  joined on run_id + window_id; window boundaries and labels are verified.
- Targets are the print-level labels nozzle_health, feeder_health and
  motion_system_health (failing = 1). One model is trained per target that has
  both classes in the selected runs; other targets are skipped with a message.
- By default only runs with the newest feature_version are used, because
  feature definitions/units differ between versions (v5: acceleration in m/s^2).

Evaluation:
- Leave-one-run-out: every run is predicted by a model that never saw any of
  its (overlapping) windows. A fold is skipped when its training runs contain
  only one class. Metrics are computed on the pooled out-of-fold predictions.
- Preprocessing (median imputation + missing-value indicators) is fitted inside
  each fold, so nothing is learned from the held-out run.
- A leakage check warns when a feature group is available for only one class,
  e.g. sensor features recorded only during failing prints.

Artifacts (models/random_forest/artifacts/):
    <target>_model.joblib, <target>_oof_predictions.csv,
    <target>_feature_importance.csv, metrics.json
"""
from __future__ import annotations

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
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT/"data"/"runs"
ARTIFACTS = Path(__file__).resolve().parent/"artifacts"
PREDICTIONS = Path(__file__).resolve().parent/"predictions"

TARGETS = ("nozzle_health", "feeder_health", "motion_system_health")
ID_COLUMNS = ("run_id", "window_id", "window_start_utc", "window_end_utc")
NON_FEATURES = {*ID_COLUMNS, *TARGETS, "processing_delay_s", "diagnostics"}
SEED = 42


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def feature_version(run_dir: Path) -> int | None:
    try:
        return json.loads((run_dir/"settings.json").read_text(encoding="utf8")).get("feature_version")
    except (OSError, json.JSONDecodeError):
        return None


def load_run(run_dir: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """One row per window: ids, labels, API features and sensor features."""
    api = pd.read_csv(run_dir/"features"/"api_features.csv")
    sensor = pd.read_csv(run_dir/"features"/"sensor_features.csv")
    api_cols = [c for c in api.columns if c not in NON_FEATURES]
    sensor_cols = [c for c in sensor.columns if c not in NON_FEATURES]

    merged = api.merge(sensor, on=["run_id", "window_id"], suffixes=("", "_sensor"),
                       validate="one_to_one")
    if len(merged) != len(api) or len(merged) != len(sensor):
        raise ValueError(f"{run_dir.name}: API and sensor feature files have different windows")
    for column in ("window_start_utc", "window_end_utc", *TARGETS):
        if not merged[column].equals(merged[column+"_sensor"]):
            raise ValueError(f"{run_dir.name}: {column} differs between API and sensor files")

    labels = pd.read_csv(run_dir/"labels"/"print_labels.csv").iloc[0]
    for target in TARGETS:
        if not (merged[target] == labels[target]).all():
            raise ValueError(f"{run_dir.name}: feature-file {target} disagrees with print_labels.csv")

    keep = [*ID_COLUMNS, *TARGETS, *api_cols, *sensor_cols]
    return merged[keep], api_cols, sensor_cols


def load_dataset(run_names: list[str] | None) -> tuple[pd.DataFrame, list[str], list[str]]:
    run_dirs = sorted(p for p in RUNS_DIR.glob("run_*") if (p/"features").is_dir())
    if run_names:
        run_dirs = [RUNS_DIR/name for name in run_names]
        versions = {feature_version(d) for d in run_dirs}
        if len(versions) > 1:
            raise ValueError(f"Selected runs mix feature versions {sorted(versions, key=str)}; "
                             "features are not comparable across versions")
    else:
        newest = max(v for v in map(feature_version, run_dirs) if v is not None)
        skipped = [d.name for d in run_dirs if feature_version(d) != newest]
        run_dirs = [d for d in run_dirs if feature_version(d) == newest]
        if skipped:
            print(f"Using feature_version {newest} runs; skipping older-format runs: {', '.join(skipped)}")

    frames, api_cols, sensor_cols = [], None, None
    for run_dir in run_dirs:
        frame, a, s = load_run(run_dir)
        if api_cols is not None and (a != api_cols or s != sensor_cols):
            raise ValueError(f"{run_dir.name}: feature columns differ from other runs")
        api_cols, sensor_cols = a, s
        frames.append(frame)
    if not frames:
        raise ValueError(f"No runs with feature files found in {RUNS_DIR}")
    return pd.concat(frames, ignore_index=True), api_cols, sensor_cols


def select_features(api_cols: list[str], sensor_cols: list[str], group: str,
                    exclude: list[str]) -> list[str]:
    columns = {"api": api_cols, "sensor": sensor_cols, "all": api_cols + sensor_cols}[group]
    unknown = set(exclude) - set(columns)
    if unknown:
        raise ValueError(f"--exclude names unknown features: {sorted(unknown)}")
    return [c for c in columns if c not in exclude]


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def make_model(args) -> Pipeline:
    return Pipeline([
        # Fitted per fold. Indicators let the forest use "feature unavailable"
        # explicitly; see the leakage check for why that needs watching.
        ("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("forest", RandomForestClassifier(
            n_estimators=args.trees,
            min_samples_leaf=args.min_leaf,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        )),
    ])


def leakage_check(data: pd.DataFrame, y: np.ndarray, api_cols, sensor_cols, features) -> list[str]:
    """Warn when a feature group is present for one class only (a data-collection shortcut)."""
    warnings = []
    for name, columns in (("API", api_cols), ("sensor", sensor_cols)):
        columns = [c for c in columns if c in features]
        if not columns:
            continue
        present = data[columns].notna().any(axis=1).to_numpy()
        rates = {label: float(present[y == value].mean()) for value, label in ((0, "healthy"), (1, "failing"))}
        if min(rates.values()) == 0 < max(rates.values()):
            only = max(rates, key=rates.get)
            warnings.append(f"{name} features exist only in {only} runs "
                            f"(healthy {rates['healthy']:.0%}, failing {rates['failing']:.0%} of windows). "
                            f"The model can separate classes by whether {name} data was recorded, "
                            f"not by component condition.")
    return warnings


def evaluate(data, X, y, groups, args) -> tuple[pd.DataFrame, dict]:
    """Leave-one-run-out out-of-fold predictions and pooled metrics."""
    oof = []
    skipped = []
    for run in pd.unique(groups):
        test = groups == run
        train = ~test
        if len(np.unique(y[train])) < 2:
            skipped.append(run)
            continue
        model = make_model(args).fit(X[train], y[train])
        score = model.predict_proba(X[test])[:, list(model.classes_).index(1)]
        oof.append(pd.DataFrame({
            "run_id": run, "window_id": data.loc[test, "window_id"].to_numpy(),
            "true": y[test], "failing_score": score,
            "predicted": (score >= args.threshold).astype(int)}))
    if not oof:
        return pd.DataFrame(), {"evaluated_runs": [], "skipped_runs": skipped}
    oof = pd.concat(oof, ignore_index=True)
    t, p = oof["true"].to_numpy(), oof["predicted"].to_numpy()
    has_both = len(np.unique(t)) == 2
    per_run = oof.groupby("run_id").agg(true=("true", "first"),
                                        fraction_predicted_failing=("predicted", "mean"),
                                        mean_failing_score=("failing_score", "mean"))
    metrics = {
        "evaluated_runs": sorted(oof["run_id"].unique().tolist()),
        "skipped_runs": skipped,
        "windows": int(len(oof)),
        "classes_in_evaluation": sorted(int(v) for v in np.unique(t)),
        "precision": float(precision_score(t, p, zero_division=0)) if has_both else None,
        "recall": float(recall_score(t, p, zero_division=0)) if 1 in t else None,
        "f1": float(f1_score(t, p, zero_division=0)) if has_both else None,
        "balanced_accuracy": float(balanced_accuracy_score(t, p)) if has_both else None,
        "confusion_matrix": confusion_matrix(t, p, labels=[0, 1]).tolist(),
        "per_run": per_run.reset_index().to_dict(orient="records"),
    }
    return oof, metrics


def print_metrics(target: str, metrics: dict) -> None:
    print(f"\n  Leave-one-run-out evaluation ({metrics['windows'] if metrics.get('windows') else 0} windows)")
    if metrics["skipped_runs"]:
        print(f"  Not evaluable (training runs would contain one class only): {', '.join(metrics['skipped_runs'])}")
    if not metrics["evaluated_runs"]:
        print("  No run could be evaluated.")
        return
    if len(metrics["classes_in_evaluation"]) < 2:
        print("  WARNING: evaluated windows contain only one class; precision, F1 and "
              "balanced accuracy are undefined. This is not evidence of real performance.")
    for key in ("precision", "recall", "f1", "balanced_accuracy"):
        if metrics[key] is not None:
            print(f"  {key:18s} {metrics[key]:.3f}")
    cm = metrics["confusion_matrix"]
    print("  Confusion matrix (rows true, cols predicted) [healthy, failing]:")
    print(f"    healthy {cm[0]}\n    failing {cm[1]}")
    print("  Per held-out run:")
    for row in metrics["per_run"]:
        truth = "failing" if row["true"] else "healthy"
        print(f"    {row['run_id']} ({truth}): {row['fraction_predicted_failing']:.0%} of windows predicted failing, "
              f"mean score {row['mean_failing_score']:.2f}")


def train(args) -> None:
    data, api_cols, sensor_cols = load_dataset(args.runs)
    features = select_features(api_cols, sensor_cols, args.features, args.exclude)
    usable = data[features].notna().any(axis=1)
    data = data[usable].reset_index(drop=True)
    X = data[features]
    groups = data["run_id"].to_numpy()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"{data['run_id'].nunique()} runs, {len(data)} windows with at least one "
          f"{args.features} feature ({len(features)} features)")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "features": features,
               "feature_group": args.features, "excluded": args.exclude,
               "threshold": args.threshold, "trees": args.trees, "min_leaf": args.min_leaf,
               "runs": sorted(data["run_id"].unique().tolist()), "targets": {}}

    for target in TARGETS:
        y = (data[target] == "failing").astype(int).to_numpy()
        runs_by_class = data.groupby(target)["run_id"].nunique().to_dict()
        print(f"\n=== {target}: {runs_by_class} runs per class")
        if len(np.unique(y)) < 2:
            print("  Skipped: every selected run has the same label, so there is nothing to learn.")
            summary["targets"][target] = {"trained": False, "reason": "single class", "runs_per_class": runs_by_class}
            continue

        warnings = leakage_check(data, y, api_cols, sensor_cols, features)
        for warning in warnings:
            print(f"  LEAKAGE WARNING: {warning}")
        oof, metrics = evaluate(data, X, y, groups, args)
        print_metrics(target, metrics)
        if not oof.empty:
            oof.to_csv(ARTIFACTS/f"{target}_oof_predictions.csv", index=False)

        model = make_model(args).fit(X, y)
        joblib.dump({"model": model, "features": features, "target": target,
                     "threshold": args.threshold}, ARTIFACTS/f"{target}_model.joblib")
        names = model.named_steps["impute"].get_feature_names_out(features)
        importance = pd.DataFrame({"feature": names,
                                   "importance": model.named_steps["forest"].feature_importances_}
                                  ).sort_values("importance", ascending=False)
        importance.to_csv(ARTIFACTS/f"{target}_feature_importance.csv", index=False)
        print("  Top features (impurity importance, final model):")
        for _, row in importance.head(8).iterrows():
            print(f"    {row.feature:45s} {row.importance:.3f}")
        summary["targets"][target] = {"trained": True, "runs_per_class": runs_by_class,
                                      "leakage_warnings": warnings, "evaluation": metrics}

    (ARTIFACTS/"metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    print(f"\nArtifacts written to {ARTIFACTS}")


def predict(args) -> None:
    models = sorted(ARTIFACTS.glob("*_model.joblib"))
    if not models:
        raise FileNotFoundError("No trained models; run the train command first")
    data, _, _ = load_dataset([args.run])
    out = data[list(ID_COLUMNS)].copy()
    for path in models:
        bundle = joblib.load(path)
        model, target = bundle["model"], bundle["target"]
        missing = set(bundle["features"]) - set(data.columns)
        if missing:
            raise ValueError(f"{args.run} lacks features used by {target} model: {sorted(missing)}")
        score = model.predict_proba(data[bundle["features"]])[:, list(model.classes_).index(1)]
        out[f"{target}_failing_score"] = score
        out[f"{target}_predicted"] = np.where(score >= bundle["threshold"], "failing", "healthy")
        print(f"{target}: {np.mean(score >= bundle['threshold']):.0%} of windows predicted failing "
              f"(mean score {score.mean():.2f}; scores are not calibrated probabilities)")
    PREDICTIONS.mkdir(parents=True, exist_ok=True)
    path = PREDICTIONS/f"{args.run}_predictions.csv"
    out.to_csv(path, index=False)
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="evaluate leave-one-run-out, then fit final models")
    p.add_argument("--runs", nargs="+", help="run folder names (default: all newest-version runs)")
    p.add_argument("--features", choices=("api", "sensor", "all"), default="all")
    p.add_argument("--exclude", nargs="+", default=[], help="feature names to drop (ablation)")
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--min-leaf", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="score every window of one run with the trained models")
    p.add_argument("--run", required=True, help="run folder name, e.g. run_0010")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
