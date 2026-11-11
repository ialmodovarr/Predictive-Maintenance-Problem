#!/usr/bin/env python3
"""XGBoost component-health classifiers trained on data/runs feature files.

Run from the project root:
    python models/xgboost/xgboost_model.py train
    python models/xgboost/xgboost_model.py train --features api
    python models/xgboost/xgboost_model.py train --features all --exclude api_extruder_temperature_error
    python models/xgboost/xgboost_model.py predict --run run_0010

Data loading, feature selection and label checks are shared with the Random
Forest script (models/random_forest/random_forest.py) so both models see
exactly the same windows, features and labels:
- API and sensor feature CSVs are joined per run on run_id + window_id.
- Targets: nozzle_health, feeder_health, motion_system_health (failing = 1).
  Targets with only one class in the selected runs are skipped.
- Only runs with the newest feature_version are used by default.

Model differences from the Random Forest:
- No imputation. XGBoost handles NaN natively by learning a default branch
  direction, so "feature missing" is still usable by the model; the coverage
  report below shows when that is a shortcut rather than a signal.
- Class imbalance is handled with balanced sample weights computed on the
  training runs of each fold.
- Shallow trees, shrinkage and row/column subsampling, because the dataset
  contains few independent runs and windows within a run are highly correlated.
- No early stopping: there are too few runs to hold out a separate validation
  run inside each fold, and stopping on the held-out run would leak.

Evaluation is leave-one-run-out, identical to the Random Forest.

Artifacts (models/xgboost/artifacts/):
    <target>_model.joblib, <target>_oof_predictions.csv,
    <target>_feature_importance.csv, metrics.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE/"artifacts"
PREDICTIONS = HERE/"predictions"
SEED = 42


def _load_shared_module():
    """Import the RF script by path for its data loaders.

    A normal package import is avoided because models/ is not a package and
    adding it to sys.path would let the models/xgboost folder shadow the
    xgboost library.
    """
    path = HERE.parent/"random_forest"/"random_forest.py"
    spec = importlib.util.spec_from_file_location("rf_shared", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _load_shared_module()
TARGETS = shared.TARGETS
ID_COLUMNS = shared.ID_COLUMNS


def make_model(args) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=args.trees,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=args.min_child_weight,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=SEED,
        n_jobs=-1,
    )


def fit_model(args, X: pd.DataFrame, y: np.ndarray) -> XGBClassifier:
    weights = compute_sample_weight("balanced", y)
    return make_model(args).fit(X, y, sample_weight=weights)


def coverage_report(data: pd.DataFrame, y: np.ndarray, api_cols, sensor_cols,
                    features) -> tuple[dict, list[str]]:
    """Share of windows per class that contain any feature from each group.

    A large difference means the model can predict the label from whether a
    data source was recorded instead of from component condition.
    """
    report, warnings = {}, []
    for name, columns in (("api", api_cols), ("sensor", sensor_cols)):
        columns = [c for c in columns if c in features]
        if not columns:
            continue
        present = data[columns].notna().any(axis=1).to_numpy()
        rates = {"healthy": float(present[y == 0].mean()), "failing": float(present[y == 1].mean())}
        report[name] = rates
        if abs(rates["healthy"] - rates["failing"]) >= 0.1:
            warnings.append(f"{name} feature coverage differs by class "
                            f"(healthy {rates['healthy']:.0%}, failing {rates['failing']:.0%} of windows); "
                            f"the model may separate classes by data availability.")
    return report, warnings


def evaluate(data, X, y, groups, args) -> tuple[pd.DataFrame, dict]:
    """Leave-one-run-out out-of-fold predictions and pooled metrics."""
    oof, skipped = [], []
    for run in pd.unique(groups):
        test = groups == run
        train = ~test
        if len(np.unique(y[train])) < 2:
            skipped.append(run)
            continue
        model = fit_model(args, X[train], y[train])
        score = model.predict_proba(X[test])[:, 1]
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


def train(args) -> None:
    data, api_cols, sensor_cols = shared.load_dataset(args.runs)
    features = shared.select_features(api_cols, sensor_cols, args.features, args.exclude)
    data = data[data[features].notna().any(axis=1)].reset_index(drop=True)
    X = data[features]
    groups = data["run_id"].to_numpy()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"{data['run_id'].nunique()} runs, {len(data)} windows with at least one "
          f"{args.features} feature ({len(features)} features)")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "model": "xgboost",
               "features": features, "feature_group": args.features, "excluded": args.exclude,
               "threshold": args.threshold, "trees": args.trees, "max_depth": args.max_depth,
               "learning_rate": args.learning_rate, "min_child_weight": args.min_child_weight,
               "seed": SEED, "runs": sorted(data["run_id"].unique().tolist()), "targets": {}}

    for target in TARGETS:
        y = (data[target] == "failing").astype(int).to_numpy()
        runs_by_class = data.groupby(target)["run_id"].nunique().to_dict()
        print(f"\n=== {target}: {runs_by_class} runs per class")
        if len(np.unique(y)) < 2:
            print("  Skipped: every selected run has the same label, so there is nothing to learn.")
            summary["targets"][target] = {"trained": False, "reason": "single class",
                                          "runs_per_class": runs_by_class}
            continue

        coverage, warnings = coverage_report(data, y, api_cols, sensor_cols, features)
        for group, rates in coverage.items():
            print(f"  {group} coverage: healthy {rates['healthy']:.0%}, failing {rates['failing']:.0%} of windows")
        for warning in warnings:
            print(f"  LEAKAGE WARNING: {warning}")

        oof, metrics = evaluate(data, X, y, groups, args)
        shared.print_metrics(target, metrics)
        if not oof.empty:
            oof.to_csv(ARTIFACTS/f"{target}_oof_predictions.csv", index=False)

        model = fit_model(args, X, y)
        joblib.dump({"model": model, "features": features, "target": target,
                     "threshold": args.threshold}, ARTIFACTS/f"{target}_model.joblib")
        gain = model.get_booster().get_score(importance_type="gain")
        importance = pd.DataFrame({"feature": features,
                                   "gain": [gain.get(f, 0.0) for f in features]})
        total = importance["gain"].sum()
        importance["gain_share"] = importance["gain"] / total if total else 0.0
        importance = importance.sort_values("gain", ascending=False)
        importance.to_csv(ARTIFACTS/f"{target}_feature_importance.csv", index=False)
        print("  Top features (gain share, final model):")
        for _, row in importance.head(8).iterrows():
            print(f"    {row.feature:45s} {row.gain_share:.3f}")
        summary["targets"][target] = {"trained": True, "runs_per_class": runs_by_class,
                                      "coverage": coverage, "leakage_warnings": warnings,
                                      "evaluation": metrics}

    (ARTIFACTS/"metrics.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf8")
    print(f"\nArtifacts written to {ARTIFACTS}")


def predict(args) -> None:
    models = sorted(ARTIFACTS.glob("*_model.joblib"))
    if not models:
        raise FileNotFoundError("No trained models; run the train command first")
    data, _, _ = shared.load_dataset([args.run])
    out = data[list(ID_COLUMNS)].copy()
    for path in models:
        bundle = joblib.load(path)
        model, target = bundle["model"], bundle["target"]
        missing = set(bundle["features"]) - set(data.columns)
        if missing:
            raise ValueError(f"{args.run} lacks features used by {target} model: {sorted(missing)}")
        score = model.predict_proba(data[bundle["features"]])[:, 1]
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
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-weight", type=float, default=5.0)
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
