#!/usr/bin/env python3
"""Nozzle-health XGBoost with run-isolated development CV and held-out testing."""
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
TARGETS = ("nozzle_health",)
ID_COLUMNS = shared.ID_COLUMNS
API_FEATURES = frozenset({
    "api_extruder_temperature_mean", "api_extruder_temperature_std",
    "api_extruder_temperature_slope", "api_extruder_temperature_error",
    "api_head_velocity_mean", "api_head_velocity_std",
})
SENSOR_FEATURES = frozenset({
    *(f"accel_{axis}_{metric}" for axis in "xyz" for metric in
      ("rms", "std", "peak_to_peak", "dominant_frequency", "kurtosis")),
    "filament_velocity_mean", "filament_movement_rate_std",
    "filament_movement_fraction_pct", "filament_longest_no_movement_interval_s",
    "thermocouple_temperature_mean", "thermocouple_temperature_std",
    "thermocouple_temperature_slope",
})


def validate_feature_columns(api: list[str], sensor: list[str]) -> None:
    """Reject undeclared predictors without changing CSV feature order."""
    for source, columns, allowed in (("API", api, API_FEATURES),
                                     ("sensor", sensor, SENSOR_FEATURES)):
        unexpected = set(columns) - allowed
        if unexpected:
            raise ValueError(f"Unexpected {source} feature columns: {sorted(unexpected)}")


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
        rates = {"healthy": float(present[y == 0].mean()) if (y == 0).any() else None, "failing": float(present[y == 1].mean()) if (y == 1).any() else None}
        report[name] = rates
        if all(v is not None for v in rates.values()) and abs(rates["healthy"] - rates["failing"]) >= 0.1:
            warnings.append(f"{name} feature coverage differs by class "
                            f"(healthy {rates['healthy']:.0%}, failing {rates['failing']:.0%} of windows); "
                            f"the model may separate classes by data availability.")
    return report, warnings


def load_data(dataset, names, labeled=True):
    """Load explicitly selected runs with matching windows and feature schemas."""
    root = shared.ROOT / "data" / dataset
    frames, schema, versions = [], None, set()
    for name in names:
        import re
        if not isinstance(name, str) or not re.fullmatch(r"run_[0-9]{4,}", name):
            raise ValueError("Invalid run identifier")
        folder = root / name
        versions.add(shared.feature_version(folder))
        if labeled:
            frame, api, sensor = shared.load_run(folder)
        else:
            a = pd.read_csv(folder / "features/api_features.csv")
            b = pd.read_csv(folder / "features/sensor_features.csv")
            api = [c for c in a if c not in shared.NON_FEATURES]
            sensor = [c for c in b if c not in shared.NON_FEATURES]
            frame = a.merge(b, on=["run_id", "window_id"], suffixes=("", "_sensor"), validate="one_to_one")
            if len(frame) != len(a) or len(frame) != len(b):
                raise ValueError("API/sensor windows differ")
            for col in ID_COLUMNS[2:]:
                if not frame[col].equals(frame[col + "_sensor"]):
                    raise ValueError("Window boundaries differ")
            if any(t in a or t in b for t in shared.TARGETS) or (folder / "labels/print_labels.csv").exists():
                frame, api, sensor = shared.load_run(folder)
        validate_feature_columns(api, sensor)
        if frame.empty or not frame.run_id.eq(name).all():
            raise ValueError(f"{name}: empty or incorrect run IDs")
        if "nozzle_health" in frame and not frame.nozzle_health.isin(["healthy", "failing"]).all():
            raise ValueError(f"{name}: invalid nozzle labels")
        if schema is not None and schema != (api, sensor):
            raise ValueError("Feature schemas differ between runs")
        schema = (api, sensor)
        frames.append(frame)
    if not frames or None in versions or len(versions) != 1:
        raise ValueError("Missing runs or incompatible feature versions")
    return pd.concat(frames, ignore_index=True), *schema, versions.pop()


def inputs(data, features):
    if not features or len(set(features)) != len(features):
        raise ValueError("Empty or duplicate feature selection")
    unexpected = set(features) - (API_FEATURES | SENSOR_FEATURES)
    if unexpected:
        raise ValueError(f"Unexpected predictive features: {sorted(unexpected)}")
    x = data[features].apply(pd.to_numeric, errors="raise")
    if np.isinf(x.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values")
    return x


def predictions(model, data, features, threshold):
    x = inputs(data, features)
    usable = x.notna().any(axis=1)
    out = data[list(ID_COLUMNS)].copy()
    if "nozzle_health" in data:
        out["true"] = data.nozzle_health.map({"healthy": 0, "failing": 1})
    out["failing_score"] = np.nan
    out["predicted"] = np.nan
    if usable.any():
        scores = model.predict_proba(x.loc[usable])[:, list(model.classes_).index(1)]
        out.loc[usable, "failing_score"] = scores
        out.loc[usable, "predicted"] = (scores >= threshold).astype(int)
    out["status"] = np.where(usable, "scored", "insufficient_data")
    return out


def metrics(oof):
    skipped = sorted(set(oof.run_id) - set(oof.loc[oof.status == "scored", "run_id"]))
    oof = oof.loc[oof.status == "scored"].copy()
    if oof.empty:
        return dict(evaluated_runs=[], skipped_runs=skipped, windows=0, classes_in_evaluation=[],
                    precision=None, recall=None, f1=None, balanced_accuracy=None,
                    confusion_matrix=[[0, 0], [0, 0]], per_run=[])
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
    return metrics



def train(args):
    from sklearn.model_selection import StratifiedKFold
    if not 0 <= args.threshold <= 1 or args.trees < 1 or args.max_depth < 1 or args.learning_rate <= 0 or args.min_child_weight < 0:
        raise ValueError("Invalid threshold or model configuration")
    art = ARTIFACTS / args.dataset
    if art.exists() and any(art.iterdir()):
        raise FileExistsError(f"Archive existing outputs before training: {art}")
    if args.dataset == "runs" and not args.split_file:
        raise ValueError("Real training requires --split-file")
    split_path = Path(args.split_file) if args.split_file else shared.ROOT / "data/mock_metadata/split.json"
    split = json.loads(split_path.read_text())
    dev, test = sorted(split["train_validation"]), sorted(split["test"])
    if not dev or not test or len(set(dev + test)) != len(dev + test):
        raise ValueError("Empty, duplicate, or overlapping split runs")
    if args.runs and sorted(args.runs) != sorted(dev + test):
        raise ValueError("--runs must match the complete predefined split")
    if args.dataset == "mock_runs":
        established = json.loads((shared.ROOT / "data/mock_metadata/split.json").read_text())
        if dev != sorted(established["train_validation"]) or test != sorted(established["test"]):
            raise ValueError("Mock split differs from established CNN split")
    data, api, sensor, version = load_data(args.dataset, dev)
    features = shared.select_features(api, sensor, args.features, args.exclude)
    x = inputs(data, features)
    usable = x.notna().any(axis=1)
    labels = data.groupby("run_id").nozzle_health.first().reindex(dev).map({"healthy": 0, "failing": 1})
    if labels.value_counts().reindex([0, 1], fill_value=0).min() < 4:
        raise ValueError("Four-fold CV requires at least four development runs per class")
    if set(data.loc[usable, "run_id"]) != set(dev):
        raise ValueError("A development run has no usable windows")
    folds, frames, fold_reports = [], [], []
    for i, (a, b) in enumerate(StratifiedKFold(4, shuffle=True, random_state=SEED).split(dev, labels)):
        tr, va = [dev[j] for j in a], [dev[j] for j in b]
        saved = HERE.parent / f"cnn_1d/experiments/full_cv_001/fold_{i}/artifacts/splits.json"
        if args.dataset == "mock_runs" and saved.exists():
            old = json.loads(saved.read_text())
            if old != dict(train=tr, validation=va, test=test):
                raise ValueError("Fold assignments differ from saved CNN experiment")
        mask = data.run_id.isin(tr) & usable
        model = fit_model(args, x.loc[mask], data.loc[mask, "nozzle_health"].eq("failing").astype(int).to_numpy())
        out = predictions(model, data.loc[data.run_id.isin(va)], features, args.threshold)
        out["fold"] = i
        frames.append(out)
        folds.append(dict(fold=i, train=tr, validation=va))
        fold_reports.append(dict(fold=i, evaluation=metrics(out)))
    y = data.nozzle_health.eq("failing").astype(int).to_numpy()
    model = fit_model(args, x.loc[usable], y[usable])
    held, _, _, test_version = load_data(args.dataset, test)
    if test_version != version:
        raise ValueError("Test feature version differs")
    test_out = predictions(model, held, features, args.threshold)
    oof = pd.concat(frames, ignore_index=True)
    coverage, warnings = coverage_report(data.loc[usable], y[usable], api, sensor, features)
    test_mask = inputs(held, features).notna().any(axis=1)
    test_coverage, test_warnings = coverage_report(held.loc[test_mask], held.loc[test_mask, "nozzle_health"].eq("failing").astype(int).to_numpy(), api, sensor, features)
    stamp = datetime.now(timezone.utc).isoformat()
    import sklearn, xgboost
    versions = dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__)
    bundle = dict(model=model, features=features, target="nozzle_health", threshold=args.threshold,
                  dataset=args.dataset, feature_version=version, training_runs=dev, created_utc=stamp, versions=versions)
    summary = dict(created_utc=stamp, model="xgboost", dataset=args.dataset, feature_version=version,
                   features=features, feature_group=args.features, excluded=args.exclude, threshold=args.threshold,
                   trees=args.trees, max_depth=args.max_depth, learning_rate=args.learning_rate,
                   min_child_weight=args.min_child_weight, seed=SEED, runs=dev, versions=versions,
                   model_parameters=json.loads(model.get_booster().save_config()),
                   splits=dict(source=str(split_path), development=dev, test=test, folds=folds), targets={})
    counts = lambda frame: dict(candidate=len(frame), scored=int(frame.status.eq("scored").sum()), insufficient_data=int(frame.status.ne("scored").sum()))
    summary["targets"]["nozzle_health"] = dict(trained=True, runs_per_class=data.groupby("nozzle_health").run_id.nunique().to_dict(),
        coverage=coverage, leakage_warnings=warnings, evaluation_protocol="four_fold_run_stratified_oof",
        evaluation=metrics(oof), fold_evaluations=fold_reports, test_evaluation=metrics(test_out),
        test_coverage=test_coverage, test_leakage_warnings=test_warnings,
        window_counts=dict(development=counts(oof), test=counts(test_out)),
        final_training=dict(runs=dev, windows=int(usable.sum()), class_weighting="balanced_window_sample_weights", artifact="nozzle_health_model.joblib"))
    art.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, art / "nozzle_health_model.joblib")
    oof.to_csv(art / "nozzle_health_oof_predictions.csv", index=False)
    test_out.to_csv(art / "nozzle_health_test_predictions.csv", index=False)
    gain = model.get_booster().get_score(importance_type="gain")
    importance = pd.DataFrame(dict(feature=features, gain=[gain.get(f, 0.0) for f in features]))
    total = importance.gain.sum()
    importance["gain_share"] = importance.gain / total if total else 0.0
    importance.sort_values("gain", ascending=False).to_csv(art / "nozzle_health_feature_importance.csv", index=False)
    (art / "metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf8")
    print(f"Saved {art}; OOF F1={metrics(oof)['f1']}; test F1={metrics(test_out)['f1']}")


def predict(args):
    bundle = joblib.load(ARTIFACTS / args.dataset / "nozzle_health_model.joblib")
    if bundle["dataset"] != args.dataset or bundle["target"] != "nozzle_health":
        raise ValueError("Incompatible model dataset or target")
    data, _, _, version = load_data(args.dataset, [args.run], labeled=False)
    if version != bundle["feature_version"]:
        raise ValueError("Incompatible feature version")
    out = predictions(bundle["model"], data, bundle["features"], bundle["threshold"])
    out = out.drop(columns="true", errors="ignore")
    out["predicted"] = out.predicted.map({0: "healthy", 1: "failing"})
    out = out.rename(columns=dict(failing_score="nozzle_health_failing_score", predicted="nozzle_health_predicted"))
    out["available_feature_count"] = inputs(data, bundle["features"]).notna().sum(axis=1)
    out["seen_in_training"] = args.run in bundle["training_runs"]
    dest = PREDICTIONS / args.dataset
    dest.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest / f"{args.run}_predictions.csv", index=False)
    print(f"Saved predictions to {dest}; scores are not calibrated probabilities")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train", help="four-fold development evaluation, final fit and test")
    p.add_argument("--runs", nargs="+", help="run folder names (must match the complete predefined split)")
    p.add_argument("--features", choices=("api", "sensor", "all"), default="all")
    p.add_argument("--exclude", nargs="+", default=[], help="feature names to drop (ablation)")
    p.add_argument("--trees", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=3)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-child-weight", type=float, default=5.0)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dataset", choices=("mock_runs", "runs"), default="mock_runs")
    p.add_argument("--split-file")
    p.set_defaults(func=train)
    p = sub.add_parser("predict", help="score every window of one run with the trained models")
    p.add_argument("--run", required=True, help="run folder name, e.g. run_0010")
    p.add_argument("--dataset", choices=("mock_runs", "runs"), default="mock_runs")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
