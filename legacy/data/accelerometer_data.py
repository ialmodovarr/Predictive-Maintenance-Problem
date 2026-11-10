"""
Preprocess print-head accelerometer data for the 3D-printer predictive-maintenance project.

Expected input columns:
    t_ms, x, y, z

Example:
    python preprocess_accelerometer.py data/accelerometer_raw.csv

Output:
    data/sensor_post_processed/accelerometer_data.csv

Assumptions to verify:
- 256 raw counts = 1 g.
- The sensor is configured for a +/-4 g full-scale range, so the raw-count
  plausibility threshold is +/-1024 counts. Change FULL_SCALE_G if needed.
- A 1-second rolling window is used. At approximately 200 Hz, that is 200 samples.
"""

from pathlib import Path
import argparse

import numpy as np
import pandas as pd


COUNTS_PER_G = 256.0
FULL_SCALE_G = 4.0
ROLLING_WINDOW_SECONDS = 1.0
EXPECTED_SAMPLE_INTERVAL_MS = 5.0  # 200 Hz; used for reporting, not resampling


def preprocess_accelerometer(input_csv: Path) -> Path:
    input_csv = input_csv.expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input CSV not found: {input_csv}\n"
            "Pass the path to your raw accelerometer CSV as an argument."
        )

    df = pd.read_csv(input_csv)

    required_columns = ["t_ms", "x", "y", "z"]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            f"Input is missing required columns: {missing_columns}. "
            f"Expected columns: {required_columns}"
        )

    # Keep original values for traceability, while making working copies numeric.
    df = df[required_columns].copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Record data-quality problems before sorting or modifying the data.
    df["timestamp_missing"] = df["t_ms"].isna()
    df["axis_value_missing"] = df[["x", "y", "z"]].isna().any(axis=1)

    # Sort by timestamp so rolling features are calculated in time order.
    df = df.sort_values("t_ms", kind="stable").reset_index(drop=True)

    # Flag duplicate timestamps. Keep the rows so the raw data is not silently lost.
    df["duplicate_timestamp"] = df["t_ms"].notna() & df["t_ms"].duplicated(keep=False)

    # Flag timestamps that do not move forward. Sorting handles out-of-order rows,
    # but duplicate timestamps are still flagged above.
    time_diff = df["t_ms"].diff()
    df["nonpositive_time_step"] = time_diff.notna() & (time_diff <= 0)

    # Plausibility threshold: +/- FULL_SCALE_G. This is not a vibration alarm
    # threshold; verify the sensor's configured full-scale range first.
    threshold_counts = COUNTS_PER_G * FULL_SCALE_G
    for axis in ["x", "y", "z"]:
        df[f"{axis}_out_of_range"] = df[axis].notna() & (
            (df[axis] < -threshold_counts) | (df[axis] > threshold_counts)
        )

    axis_out_of_range_columns = ["x_out_of_range", "y_out_of_range", "z_out_of_range"]
    df["any_axis_out_of_range"] = df[axis_out_of_range_columns].any(axis=1)

    # Create cleaned working signals. Out-of-range samples become NaN for derived
    # features, but the original raw-count columns remain unchanged.
    for axis in ["x", "y", "z"]:
        valid_axis = df[axis].where(~df[f"{axis}_out_of_range"])
        df[f"{axis}_g"] = valid_axis / COUNTS_PER_G

    # Approximate sampling rate from positive timestamp intervals.
    positive_steps = df["t_ms"].diff()
    positive_steps = positive_steps[positive_steps > 0]
    if not positive_steps.empty:
        median_step_ms = float(positive_steps.median())
        estimated_rate_hz = 1000.0 / median_step_ms
        rolling_window_samples = max(
            2, int(round(ROLLING_WINDOW_SECONDS * estimated_rate_hz))
        )