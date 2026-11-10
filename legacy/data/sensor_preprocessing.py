"""
Live sensor preprocessing + feature engineering for the 3D-printer
predictive-maintenance capstone.

The script:
- Selects the newest print_*.csv file by modification time.
- Detects when a newer timestamped input file appears.
- Reads newly appended rows from the active file.
- Builds non-overlapping time windows (default: 1 second).
- Calculates filament, temperature, and accelerometer features.
- Appends one feature row per completed window to the output CSV.

Configured for the latest supplied sensor logger CSV:
    time_s, accel_x_g, accel_y_g, accel_z_g, filament_mm,
    hotend_temp_C, board_t_us, sample_index, accel_x_counts,
    accel_y_counts, accel_z_counts, filament_counts

Optional columns that enable error calculations (not present in the
latest supplied sample):
    filament_commanded_mm
    hotend_target_temp_C

Install:
    pip install pandas numpy

SciPy is not required; dominant frequency is calculated with NumPy.
"""

from __future__ import annotations

import csv
import io
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

# Resolve paths relative to this script, so it works even if launched
# from a different current working directory. Keep this script in the
# Predictive-Maintenance-Problem project root.
PROJECT_ROOT = Path(__file__).resolve().parent
SENSOR_DIR = Path("sensors")  # Absolute input directory requested.
OUTPUT_DIR =  Path("sensor_preprocessed")
OUTPUT_FILE: Optional[Path] = None  # Set to a unique file when the script starts.

INPUT_GLOB = "print_*.csv"

WINDOW_SECONDS = 1.0
POLL_INTERVAL_SECONDS = 0.25

# Your sensor logger is approximately 800 Hz. The actual sample rate
# is estimated from time_s for each window rather than assumed fixed.
EXPECTED_ACCEL_HZ = 800.0

# Minimum number of accelerometer samples required to calculate
# frequency-domain features for a window.
MIN_ACCEL_SAMPLES = 32

# Set True to print a status line whenever a feature window is written.
PRINT_STATUS = True


# ---------------------------------------------------------------------
# COLUMN NAMES
# ---------------------------------------------------------------------

TIME_COL = "time_s"
BOARD_TIME_COL = "board_t_us"
SAMPLE_INDEX_COL = "sample_index"

ACCEL_COLUMNS = ["accel_x_g", "accel_y_g", "accel_z_g"]
ACCEL_COUNT_COLUMNS = ["accel_x_counts", "accel_y_counts", "accel_z_counts"]
FILAMENT_COUNTS_COL = "filament_counts"

FILAMENT_ACTUAL_COL = "filament_mm"
FILAMENT_COMMAND_COL = "filament_commanded_mm"

TEMP_CURRENT_COL = "hotend_temp_C"
TEMP_TARGET_COL = "hotend_target_temp_C"


# ---------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------

def newest_sensor_file() -> Optional[Path]:
    """Return the newest timestamped print CSV in SENSOR_DIR."""
    files = list(SENSOR_DIR.glob(INPUT_GLOB))

    if not files:
        return None

    return max(files, key=lambda p: p.stat().st_mtime)


def safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric, replacing invalid values with NaN."""
    return pd.to_numeric(series, errors="coerce")


def last_valid(series: pd.Series) -> float:
    """Return the last valid numeric value, or NaN."""
    values = safe_numeric(series).dropna()
    if values.empty:
        return np.nan
    return float(values.iloc[-1])


def mean_valid(series: pd.Series) -> float:
    """Return the mean of valid numeric values, or NaN."""
    values = safe_numeric(series).dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def std_valid(series: pd.Series) -> float:
    """Return sample standard deviation, or NaN if insufficient data."""
    values = safe_numeric(series).dropna()
    if len(values) < 2:
        return np.nan
    return float(values.std(ddof=1))


def estimate_sampling_rate(time_values: np.ndarray) -> float:
    """Estimate sampling rate from the median positive time difference."""
    t = np.asarray(time_values, dtype=float)
    t = t[np.isfinite(t)]

    if len(t) < 2:
        return np.nan

    differences = np.diff(t)
    differences = differences[differences > 0]

    if len(differences) == 0:
        return np.nan

    return float(1.0 / np.median(differences))


def dominant_frequency(time_values: np.ndarray,
                       vibration_values: np.ndarray) -> float:
    """
    Estimate the strongest non-DC frequency in Hz.

    The input timestamps are not perfectly uniform, so the signal is
    interpolated onto a uniform time grid before applying an FFT.
    """
    t = np.asarray(time_values, dtype=float)
    x = np.asarray(vibration_values, dtype=float)

    valid = np.isfinite(t) & np.isfinite(x)
    t = t[valid]
    x = x[valid]

    if len(t) < MIN_ACCEL_SAMPLES:
        return np.nan

    order = np.argsort(t)
    t = t[order]
    x = x[order]

    # Remove duplicate timestamps.
    t_unique, unique_indices = np.unique(t, return_index=True)
    x = x[unique_indices]

    if len(t_unique) < MIN_ACCEL_SAMPLES:
        return np.nan

    duration = t_unique[-1] - t_unique[0]
    if duration <= 0:
        return np.nan

    fs = estimate_sampling_rate(t_unique)
    if not np.isfinite(fs) or fs <= 0:
        return np.nan

    # Uniform sampling grid based on the measured rate.
    sample_count = max(MIN_ACCEL_SAMPLES, int(round(duration * fs)) + 1)
    uniform_t = np.linspace(t_unique[0], t_unique[-1], sample_count)
    uniform_x = np.interp(uniform_t, t_unique, x)

    # Remove DC and apply a Hann window to reduce spectral leakage.
    uniform_x = uniform_x - np.mean(uniform_x)
    window = np.hanning(len(uniform_x))

    spectrum = np.fft.rfft(uniform_x * window)
    frequencies = np.fft.rfftfreq(len(uniform_x), d=1.0 / fs)
    amplitudes = np.abs(spectrum)

    if len(amplitudes) <= 1:
        return np.nan

    # Ignore the zero-frequency/DC component.
    amplitudes[0] = 0.0
    peak_index = int(np.argmax(amplitudes))

    return float(frequencies[peak_index])


# ---------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------

def engineer_window_features(window: pd.DataFrame,
                             input_filename: str,
                             window_start_s: float,
                             window_end_s: float) -> dict:
    """Calculate all features for one completed time window."""

    features = {
        "source_file": input_filename,
        "window_start_s": window_start_s,
        "window_end_s": window_end_s,
        "window_duration_s": window_end_s - window_start_s,
        "sample_count": len(window),
    }

    # -------------------------------------------------------------
    # 1. FILAMENT MOTION
    # -------------------------------------------------------------

    if FILAMENT_ACTUAL_COL in window.columns:
        filament = safe_numeric(window[FILAMENT_ACTUAL_COL]).dropna()

        # Cumulative total extruded at the end of this window.
        features["total_filament_extruded_mm"] = (
            float(filament.iloc[-1]) if not filament.empty else np.nan
        )

        # Amount extruded during this window.
        features["filament_extruded_window_mm"] = (
            float(filament.iloc[-1] - filament.iloc[0])
            if len(filament) >= 2 else np.nan
        )
    else:
        features["total_filament_extruded_mm"] = np.nan
        features["filament_extruded_window_mm"] = np.nan

    # Filament error = commanded movement - actual movement.
    # Requires a commanded cumulative filament column.
    if (FILAMENT_COMMAND_COL in window.columns
            and FILAMENT_ACTUAL_COL in window.columns):
        commanded = safe_numeric(window[FILAMENT_COMMAND_COL]).dropna()
        actual = safe_numeric(window[FILAMENT_ACTUAL_COL]).dropna()

        if len(commanded) >= 1 and len(actual) >= 1:
            features["filament_error_mm"] = float(
                commanded.iloc[-1] - actual.iloc[-1]
            )
        else:
            features["filament_error_mm"] = np.nan
    else:
        features["filament_error_mm"] = np.nan

    # -------------------------------------------------------------
    # 2. TEMPERATURE
    # -------------------------------------------------------------

    if TEMP_CURRENT_COL in window.columns:
        temp = safe_numeric(window[TEMP_CURRENT_COL])
        features["temperature_current_C"] = last_valid(temp)
        features["temperature_mean_C"] = mean_valid(temp)
        features["temperature_std_C"] = std_valid(temp)
        features["temperature_valid_samples"] = int(temp.notna().sum())
    else:
        features["temperature_current_C"] = np.nan
        features["temperature_mean_C"] = np.nan
        features["temperature_std_C"] = np.nan
        features["temperature_valid_samples"] = 0

    # Temperature error = target temperature - current temperature.
    if (TEMP_TARGET_COL in window.columns
            and TEMP_CURRENT_COL in window.columns):
        current = safe_numeric(window[TEMP_CURRENT_COL])
        target = safe_numeric(window[TEMP_TARGET_COL])
        paired = pd.concat(
            [current.rename("current"), target.rename("target")],
            axis=1
        ).dropna()

        features["temperature_error_C"] = (
            float(paired["target"].iloc[-1] - paired["current"].iloc[-1])
            if not paired.empty else np.nan
        )
    else:
        features["temperature_error_C"] = np.nan

    # -------------------------------------------------------------
    # 3. ACCELEROMETER
    # -------------------------------------------------------------

    if all(column in window.columns for column in ACCEL_COLUMNS):
        accel = window[ACCEL_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if len(accel) >= 2:
            ax = accel["accel_x_g"].to_numpy(dtype=float)
            ay = accel["accel_y_g"].to_numpy(dtype=float)
            az = accel["accel_z_g"].to_numpy(dtype=float)

            # Preserve raw axis summaries as useful RF features.
            features["accel_x_mean_g"] = float(np.mean(ax))
            features["accel_y_mean_g"] = float(np.mean(ay))
            features["accel_z_mean_g"] = float(np.mean(az))

            features["accel_x_std_g"] = float(np.std(ax, ddof=1))
            features["accel_y_std_g"] = float(np.std(ay, ddof=1))
            features["accel_z_std_g"] = float(np.std(az, ddof=1))

            # Remove each axis's window mean to reduce the effect of
            # gravity and static sensor orientation.
            ax_dynamic = ax - np.mean(ax)
            ay_dynamic = ay - np.mean(ay)
            az_dynamic = az - np.mean(az)

            # Dynamic 3-axis vibration magnitude.
            vibration_magnitude = np.sqrt(
                ax_dynamic**2 + ay_dynamic**2 + az_dynamic**2
            )

            # Overall vibration RMS.
            features["vibration_rms_g"] = float(
                np.sqrt(np.mean(vibration_magnitude**2))
            )

            # Variability of the dynamic vibration magnitude.
            features["vibration_std_g"] = float(
                np.std(vibration_magnitude, ddof=1)
            )

            # Largest dynamic vibration magnitude in the window.
            features["peak_acceleration_g"] = float(
                np.max(vibration_magnitude)
            )

            # Range of dynamic vibration magnitude.
            features["peak_to_peak_g"] = float(
                np.max(vibration_magnitude) - np.min(vibration_magnitude)
            )

            # Frequency of strongest vibration component.
            # board_t_us is the logger's microsecond clock. Prefer it for
            # sample-rate/frequency calculations because it is the closest
            # representation of the sensor acquisition timing. Fall back
            # to time_s if the board clock is absent or invalid.
            if BOARD_TIME_COL in window.columns:
                board_time_us = safe_numeric(
                    window.loc[accel.index, BOARD_TIME_COL]
                )
                time_values = board_time_us / 1_000_000.0
                if time_values.isna().any() or time_values.nunique() < 2:
                    time_values = safe_numeric(
                        window.loc[accel.index, TIME_COL]
                    )
            else:
                time_values = safe_numeric(
                    window.loc[accel.index, TIME_COL]
                )

            time_array = time_values.to_numpy(dtype=float)
            features["dominant_frequency_hz"] = dominant_frequency(
                time_array, vibration_magnitude
            )
            features["estimated_accel_sampling_rate_hz"] = (
                estimate_sampling_rate(time_array)
            )
            features["accelerometer_valid_samples"] = len(accel)

            # Preserve logger diagnostics in the feature table. These are
            # useful for checking dropped samples and sensor conversion.
            if SAMPLE_INDEX_COL in window.columns:
                sample_idx = safe_numeric(window.loc[accel.index, SAMPLE_INDEX_COL]).dropna()
                features["sample_index_start"] = (
                    int(sample_idx.iloc[0]) if not sample_idx.empty else np.nan
                )
                features["sample_index_end"] = (
                    int(sample_idx.iloc[-1]) if not sample_idx.empty else np.nan
                )
            if BOARD_TIME_COL in window.columns:
                board_times = safe_numeric(window.loc[accel.index, BOARD_TIME_COL]).dropna()
                features["board_time_start_us"] = (
                    int(board_times.iloc[0]) if not board_times.empty else np.nan
                )
                features["board_time_end_us"] = (
                    int(board_times.iloc[-1]) if not board_times.empty else np.nan
                )
            for count_col in ACCEL_COUNT_COLUMNS:
                if count_col in window.columns:
                    features[f"{count_col}_mean"] = mean_valid(window.loc[accel.index, count_col])
            if FILAMENT_COUNTS_COL in window.columns:
                features["filament_counts_end"] = last_valid(window[FILAMENT_COUNTS_COL])

        else:
            for column in [
                "accel_x_mean_g", "accel_y_mean_g", "accel_z_mean_g",
                "accel_x_std_g", "accel_y_std_g", "accel_z_std_g",
                "vibration_rms_g", "vibration_std_g",
                "peak_acceleration_g", "peak_to_peak_g",
                "dominant_frequency_hz",
                "estimated_accel_sampling_rate_hz",
            ]:
                features[column] = np.nan

            features["accelerometer_valid_samples"] = len(accel)
    else:
        features["accelerometer_valid_samples"] = 0

    return features


# ---------------------------------------------------------------------
# LIVE CSV READER
# ---------------------------------------------------------------------

def read_complete_csv(input_file: Path) -> pd.DataFrame:
    """
    Read only newline-terminated CSV content.

    This prevents the live reader from consuming a row while the sensor
    logger is still writing it.
    """
    try:
        raw = input_file.read_bytes()
    except OSError:
        return pd.DataFrame()

    last_newline = raw.rfind(b"\n")
    if last_newline < 0:
        return pd.DataFrame()

    complete_bytes = raw[:last_newline + 1]

    try:
        return pd.read_csv(io.BytesIO(complete_bytes))
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return pd.DataFrame()


def read_new_rows(input_file: Path, rows_already_read: int) -> pd.DataFrame:
    """Return complete CSV rows appended since the previous poll."""
    data = read_complete_csv(input_file)

    if data.empty or len(data) <= rows_already_read:
        return pd.DataFrame()

    new_data = data.iloc[rows_already_read:].copy()

    for column in [
        TIME_COL,
        BOARD_TIME_COL,
        SAMPLE_INDEX_COL,
        *ACCEL_COLUMNS,
        *ACCEL_COUNT_COLUMNS,
        FILAMENT_COUNTS_COL,
        FILAMENT_ACTUAL_COL,
        FILAMENT_COMMAND_COL,
        TEMP_CURRENT_COL,
        TEMP_TARGET_COL,
    ]:
        if column in new_data.columns:
            new_data[column] = pd.to_numeric(
                new_data[column], errors="coerce"
            )

    if TIME_COL not in new_data.columns:
        raise ValueError(f"Input CSV is missing required column: {TIME_COL}")

    return new_data.dropna(subset=[TIME_COL])


def append_feature_row(features: dict) -> None:
    """Append a feature row to the output CSV, writing the header once."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if OUTPUT_FILE is None:
        raise RuntimeError("Output file has not been initialized. Start run_live_pipeline() first.")

    output_exists = OUTPUT_FILE.exists() and OUTPUT_FILE.stat().st_size > 0

    pd.DataFrame([features]).to_csv(
        OUTPUT_FILE,
        mode="a",
        header=not output_exists,
        index=False
    )


# ---------------------------------------------------------------------
# MAIN LIVE LOOP
# ---------------------------------------------------------------------

def run_live_pipeline() -> None:
    global OUTPUT_FILE

    SENSOR_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a unique output file for this execution. Feature rows from
    # this run are appended to this file only; later runs get a new file.
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    OUTPUT_FILE = OUTPUT_DIR / f"sensor_features_{run_timestamp}.csv"
    OUTPUT_FILE.touch(exist_ok=False)

    active_file = None
    rows_already_read = 0

    pending_data = pd.DataFrame()
    next_window_start = None

    print("=" * 65)
    print("LIVE SENSOR PREPROCESSING + FEATURE ENGINEERING")
    print(f"Watching: {SENSOR_DIR.resolve()}")
    print(f"Output:   {OUTPUT_FILE.resolve()}")
    print(f"Window:   {WINDOW_SECONDS:.2f} seconds")
    print("Press Ctrl+C to stop.")
    print("=" * 65)

    try:
        while True:
            newest_file = newest_sensor_file()

            if newest_file is None:
                print("Waiting for a print_*.csv file...")
                time.sleep(POLL_INTERVAL_SECONDS)
                continue

            # Switch to a newer timestamped sensor file.
            if active_file != newest_file:
                active_file = newest_file
                rows_already_read = 0
                pending_data = pd.DataFrame()
                next_window_start = None

                print(f"\nMonitoring input file: {active_file.name}")

            # Read only complete, newline-terminated rows.
            current_data = read_complete_csv(active_file)

            # Detect a file that has been truncated/restarted.
            if len(current_data) < rows_already_read:
                rows_already_read = 0
                pending_data = pd.DataFrame()
                next_window_start = None

            new_rows = read_new_rows(active_file, rows_already_read)
            rows_already_read = len(current_data)

            if not new_rows.empty:
                if pending_data.empty:
                    pending_data = new_rows.copy()
                else:
                    pending_data = pd.concat(
                        [pending_data, new_rows], ignore_index=True
                    )

                pending_data = pending_data.sort_values(TIME_COL)
                pending_data = pending_data.reset_index(drop=True)

                if next_window_start is None:
                    next_window_start = float(pending_data[TIME_COL].iloc[0])

            # Process every fully available window.
            while (
                not pending_data.empty
                and next_window_start is not None
            ):
                window_end = next_window_start + WINDOW_SECONDS

                # Wait until the input contains data at/after window_end.
                if float(pending_data[TIME_COL].max()) < window_end:
                    break

                window = pending_data[
                    (pending_data[TIME_COL] >= next_window_start)
                    & (pending_data[TIME_COL] < window_end)
                ].copy()

                if not window.empty:
                    features = engineer_window_features(
                        window=window,
                        input_filename=active_file.name,
                        window_start_s=next_window_start,
                        window_end_s=window_end
                    )

                    append_feature_row(features)

                    if PRINT_STATUS:
                        rms = features.get("vibration_rms_g", np.nan)
                        peak = features.get("peak_acceleration_g", np.nan)
                        filament = features.get(
                            "total_filament_extruded_mm", np.nan
                        )
                        temp = features.get("temperature_current_C", np.nan)

                        temp_valid = features.get("temperature_valid_samples", 0)
                        temp_text = (f"{temp:.2f} C" if np.isfinite(temp) else "n/a")
                        print(
                            f"Window {next_window_start:.2f}-{window_end:.2f}s | "
                            f"Samples: {len(window):4d} | "
                            f"Vibration RMS: {rms:.4f} g | "
                            f"Peak: {peak:.4f} g | "
                            f"Filament: {filament:.3f} mm | "
                            f"Temp: {temp_text} ({temp_valid} valid)"
                        )

                next_window_start = window_end

                # Discard rows that are no longer needed.
                pending_data = pending_data[
                    pending_data[TIME_COL] >= next_window_start
                ].reset_index(drop=True)

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nStopping live preprocessing pipeline...")
        print(f"Feature output saved to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    run_live_pipeline()
