"""
Post-process Ultimaker S3 API logs for predictive-maintenance features.

Expected project layout:
data/
    process_api_data.py        <- this script
    api_data/
        api_data.txt            <- raw log written by the API collector
    processed/
        printer_features.csv
        rejected_values.csv
        processing_summary.txt

Install dependencies:
    pip install pandas numpy

Important:
- This parser reads the text log format produced by the collector:
  endpoint divider, endpoint name, "Laptop timestamp: ...", JSON payload.
- It processes /api/v1/printer snapshots only.
- Implausible individual measurements are set to NaN rather than dropping
  the entire snapshot. This preserves valid measurements from that timestamp.
- Thresholds below are broad engineering sanity checks, not official
  Ultimaker firmware limits. Tune them after checking the printer/API docs.
"""

from __future__ import annotations
import hashlib
import requests
import os
import json
import secrets
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import time
import numpy as np
import pandas as pd

PRINTER = "http://172.27.135.1"

USERNAME = "6bf926c99944ad9e15da49c2ed1bffb5"
PASSWORD = "716b548893f2fe16f0d1929b94857484d291b8af67e61685ba9c1ad24f682e30"

# ---------------------------------------------------------------------------
# Paths: this script is in data/, and the raw log is in data/api_data/
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_FILE = SCRIPT_DIR / "api_data" / "api_data.txt"
OUTPUT_DIR = SCRIPT_DIR / "processed"

OUTPUT_CSV = OUTPUT_DIR / "printer_features.csv"
RF_FEATURES_CSV = OUTPUT_DIR / "random_forest_features.csv"
RF_METADATA_CSV = OUTPUT_DIR / "random_forest_metadata.csv"
REJECTED_CSV = OUTPUT_DIR / "rejected_values.csv"
SUMMARY_FILE = OUTPUT_DIR / "processing_summary.txt"

# Rolling window: 5 samples. At ~1 API sample/sec, this is approximately 5 sec.
ROLLING_WINDOW = 5
MIN_PERIODS = 2

PRINTER_URL = os.getenv("PRINTER_URL", "http://192.168.1.100")
MAX_EXTRUDER_TEMP_C = 290.0

# request to send commands to printer
def digest_put(uri, payload):
    """Send an authenticated PUT request to the printer API."""

    # First, try the request without authentication.
    # A protected endpoint should respond with 401 and a Digest challenge.
    response = requests.put(
        PRINTER + uri,
        json=payload,
        timeout=5
    )

    if response.ok:
        return response

    if response.status_code != 401:
        response.raise_for_status()

    # Read the printer's authentication challenge.
    challenge = response.headers["WWW-Authenticate"]

    realm = re.search(r'realm="([^"]+)"', challenge).group(1)
    nonce = re.search(r'nonce="([^"]+)"', challenge).group(1)
    qop = re.search(r'qop="([^"]+)"', challenge).group(1)

    # Create the Digest authentication values.
    nc = "00000001"
    cnonce = secrets.token_hex(16)

    HA1 = hashlib.sha256(
        f"{USERNAME}:{realm}:{PASSWORD}".encode()
    ).hexdigest()

    # IMPORTANT: Use PUT here, not GET.
    HA2 = hashlib.sha256(
        f"PUT:{uri}".encode()
    ).hexdigest()

    response_hash = hashlib.sha256(
        f"{HA1}:{nonce}:{nc}:{cnonce}:{qop}:{HA2}".encode()
    ).hexdigest()

    authorization = (
        f'Digest username="{USERNAME}", '
        f'realm="{realm}", '
        f'nonce="{nonce}", '
        f'uri="{uri}", '
        f'algorithm=SHA-256, '
        f'response="{response_hash}", '
        f'qop={qop}, '
        f'nc={nc}, '
        f'cnonce="{cnonce}"'
    )

    # Retry the PUT request with authentication.
    response = requests.put(
        PRINTER + uri,
        json=payload,
        headers={"Authorization": authorization},
        timeout=5
    )

    response.raise_for_status()
    return response

# stops printer for extreme extruder values 
# def stop_printer():
#     """Attempt to abort printing and turn off extruder 1 heating."""

#     # Command 1: Abort the current print job.
#     abort_uri = "/api/v1/print_job/state"

#     # Command 2: Set extruder 1's hotend target to 0 degrees C.
#     temperature_uri = (
#         "/api/v1/printer/heads/0/extruders/0/"
#         "hotend/temperature/target"
#     )

#     try:
#         response = digest_put(
#             abort_uri,
#             {"target": "abort"}
#         )
#         print(f"Print-abort request returned HTTP {response.status_code}.")

#         response = digest_put(
#             temperature_uri,
#             0.0
#         )
#         print(
#             f"Hotend target request returned HTTP {response.status_code}."
#         )

#     except requests.RequestException as exc:
#         print(f"CRITICAL: Printer stop command failed: {exc}")
#         print("Check the printer directly and stop it manually if needed.")

# Broad sanity bounds. These are intentionally permissive and should be
# calibrated against verified API documentation and controlled experiments.
LIMITS = {

    # Temperatures, degrees C
    #max extruder temperature is 280 degrees, so limit set accordingly
    "extruder1_temp_current": (0.0, 290.0),
    "extruder1_temp_target": (0.0, 290.0),

    "extruder1_material_extruded": (0.0, 1e12),
    "extruder1_max_temperature_exposed_c": (0.0, 350.0),
    "extruder1_prints_since_cleaned": (0.0, 1e7),
    "extruder1_time_spent_hot_s": (0.0, 1e10),

    #max bed temperature is 140, so limit set accordingly
    "bed_temp_current": (0.0, 150.0),
    "bed_temp_target": (0.0, 150.0),

    # Printer-head position, mm; broad S3 build-envelope checks
    "head_x_mm": (0.0, 230.0),
    "head_y_mm": (0.0, 190.0),
    "head_z_mm": (0.0, 200.0),

    # Motion settings. Broad nonnegative plausibility bounds
    "extruder1_jerk": (0.0, 1000.0),
    "extruder1_acceleration": (0.0, 20000.0),
    "extruder1_max_speed": (0.0, 1000.0),
    "head_x_jerk": (0.0, 1000.0),
    "head_y_jerk": (0.0, 1000.0),
    "head_z_jerk": (0.0, 1000.0),
    "head_acceleration": (0.0, 20000.0),
    "head_x_max_speed": (0.0, 1000.0),
    "head_y_max_speed": (0.0, 1000.0),
    "head_z_max_speed": (0.0, 1000.0),
}

# The API snapshot contains two extruder objects. Extruder 1 here means index 0
# in heads[0]["extruders"], which is the active 200 C AA 0.4 hotend in the
# supplied sample. Extruder 2 is deliberately not included as a feature.


# Read the entire API log as text; replace undecodable characters instead of failing.
def parse_log(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Extract JSON snapshots from the raw text log."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find raw API log: {path}\n"
            "Expected api_data/api_data.txt next to this script's data folder."
        )
    #replace undecodable characters with ? instead of failing.
    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    # In the collector's format, the endpoint is between two divider lines;
    # the timestamp and JSON payload follow the second divider.
    pattern = re.compile(
        r"(?P<endpoint>/api/v1/[^\s]+)\s*\n\s*={10,}\s*\n"
        r"(?P<body>.*?)(?=\n\s*={10,}\s*\n|\Z)",
        re.DOTALL,
    )

    for section_number, match in enumerate(pattern.finditer(text), start=1):
        endpoint = match.group("endpoint").strip()
        if endpoint != "/api/v1/printer":
            continue

        body = match.group("body")
        timestamp_match = re.search(r"Laptop timestamp:\s*([^\r\n]+)", body)
        if not timestamp_match:
            errors.append(f"Printer section {section_number}: missing laptop timestamp")
            continue

        json_start = body.find("{", timestamp_match.end())
        if json_start < 0:
            errors.append(f"Printer section {section_number}: missing JSON payload")
            continue

        try:
            payload, _ = json.JSONDecoder().raw_decode(body[json_start:].lstrip())
        except json.JSONDecodeError as exc:
            errors.append(f"Printer section {section_number}: invalid JSON ({exc})")
            continue

        records.append({
            "laptop_time": timestamp_match.group(1).strip(),
            "payload": payload,
            "section": section_number,
        })

    return records, errors

def parse_clock_times(clock_strings: list[str]) -> pd.Series:
    """
    Convert laptop HH:MM:SS.mmm strings into monotonically increasing seconds.
    The raw log has time-of-day only, so midnight rollover is inferred.
    """
    seconds = []
    day_offset = 0
    previous = None

    for value in clock_strings:
        try:
            t = datetime.strptime(value, "%H:%M:%S.%f")
        except ValueError:
            try:
                t = datetime.strptime(value, "%H:%M:%S")
            except ValueError:
                seconds.append(np.nan)
                continue

        sec = (
            t.hour * 3600
            + t.minute * 60
            + t.second
            + t.microsecond / 1_000_000
        )

        # Treat a large backward jump as midnight rollover.
        if previous is not None and sec < previous - 12 * 3600:
            day_offset += 24 * 3600

        seconds.append(sec + day_offset)
        previous = sec

    series = pd.Series(seconds, dtype="float64")
    valid = series.dropna()
    if not valid.empty:
        series = series - valid.iloc[0]
    return series

# Define a safe helper for reading fields several levels deep in nested JSON.
def get_nested(obj: dict, *keys, default=np.nan):
    """Safely retrieve a nested dictionary value."""
    value: Any = obj
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value

# Define a helper that converts API values to finite numbers.
def numeric(value):
    """Convert a value to float; invalid/sentinel values become NaN."""
    if value is None or isinstance(value, bool):
        return np.nan
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result if np.isfinite(result) else np.nan

# Define a function that converts one nested printer snapshot into a flat feature row.
def extract_row(record: dict[str, Any]) -> dict[str, Any]:
    """Flatten the API fields requested for the feature pipeline."""
    # Retrieve the decoded printer JSON from the parser record.
    payload = record["payload"]
    # Get the list of print heads, using an empty list if absent.
    heads = payload.get("heads") or []
    # Use the first head if it exists and is a dictionary; otherwise use an empty dictionary.
    head = heads[0] if len(heads) > 0 and isinstance(heads[0], dict) else {}
    # Get extruders belonging to the first print head.
    extruders = head.get("extruders") or []
    extruder1 = (
        extruders[0]
        if len(extruders) > 0 and isinstance(extruders[0], dict)
        else {}
    )

    # Get the first extruder feeder settings, or an empty dictionary if missing.
    feeder = extruder1.get("feeder") or {}
    # Get the first extruder hotend information.
    hotend = extruder1.get("hotend") or {}
    # Get current and target hotend temperature fields.
    hotend_temp = hotend.get("temperature") or {}
    # Safely retrieve the print-bed temperature dictionary.
    bed_temp = get_nested(payload, "bed", "temperature", default={})
    # Ensure the bed temperature value is a dictionary before reading fields from it.
    if not isinstance(bed_temp, dict):
        bed_temp = {}

    # Get current X, Y, and Z print-head positions.
    position = head.get("position") or {}
    # Get X, Y, and Z jerk settings for the print head.
    jerk = head.get("jerk") or {}
    # Get the maximum speed setting for each print-head axis.
    max_speed = head.get("max_speed") or {}

    # Build one flat row whose keys become feature-column names in the CSV.
    row = {
        "laptop_time": record["laptop_time"],
        "section": record["section"],

        # Extruder 1 / feeder
        "extruder1_jerk": numeric(feeder.get("jerk")),
        "extruder1_acceleration": numeric(feeder.get("acceleration")),
        "extruder1_max_speed": numeric(feeder.get("max_speed")),
        "extruder1_temp_current": numeric(hotend_temp.get("current")),
        "extruder1_temp_target": numeric(hotend_temp.get("target")),

        # Extruder 1 / hotend lifetime statistics from the printer API
        "extruder1_material_extruded": numeric(
            get_nested(extruder1, "hotend", "statistics", "material_extruded")
        ),
        "extruder1_max_temperature_exposed_c": numeric(
            get_nested(extruder1, "hotend", "statistics", "max_temperature_exposed")
        ),
        "extruder1_prints_since_cleaned": numeric(
            get_nested(extruder1, "hotend", "statistics", "prints_since_cleaned")
        ),
        "extruder1_time_spent_hot_s": numeric(
            get_nested(extruder1, "hotend", "statistics", "time_spent_hot")
        ),

        # Print head / motion
        "head_x_mm": numeric(position.get("x")),
        "head_y_mm": numeric(position.get("y")),
        "head_z_mm": numeric(position.get("z")),
        "head_x_jerk": numeric(jerk.get("x")),
        "head_y_jerk": numeric(jerk.get("y")),
        "head_z_jerk": numeric(jerk.get("z")),
        "head_x_max_speed": numeric(max_speed.get("x")),
        "head_y_max_speed": numeric(max_speed.get("y")),
        "head_z_max_speed": numeric(max_speed.get("z")),
        "head_acceleration": numeric(head.get("acceleration")),

        # Print bed
        "bed_temp_current": numeric(bed_temp.get("current")),
        "bed_temp_target": numeric(bed_temp.get("target")),
    }
    # Give the extracted row back to the caller.
    return row


# Define a function that masks out-of-range measurements without deleting the whole snapshot.
def apply_thresholds(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply engineering sanity checks per metric.
    Bad values are masked as NaN and recorded in a rejection audit file.
    Rows are retained so one bad field does not erase other valid sensors.
    """
    # Work on a copy so the original DataFrame is not modified in place.
    clean = df.copy()
    # Prepare a list describing every value that fails its sanity check.
    rejected: list[dict[str, Any]] = []

    # Check each configured feature against its minimum and maximum allowed values.
    for column, (low, high) in LIMITS.items():
        # Skip configured features that are not present in this particular table.
        if column not in clean:
            continue

        # Convert the column to numbers and turn unparseable entries into missing values.
        original = pd.to_numeric(clean[column], errors="coerce")
        # Flag numeric values below the minimum or above the maximum.
        invalid = original.notna() & ((original < low) | (original > high))

        # Visit each row containing a flagged value so its details can be logged.
        for idx in clean.index[invalid]:
            rejected.append({
                "row_index": int(idx),
                "laptop_time": clean.at[idx, "laptop_time"],
                "metric": column,
                "rejected_value": original.at[idx],
                "allowed_min": low,
                "allowed_max": high,
                "reason": "outside broad engineering sanity bounds",
            })

        # Replace bad measurements with NaN but retain the rest of each snapshot.
        clean.loc[invalid, column] = np.nan
        # Ensure the cleaned column has a numeric dtype.
        clean[column] = pd.to_numeric(clean[column], errors="coerce")

    # Zero is allowed for temperature targets (e.g., a hotend switched off).
    # Negative temperature values are already masked by the limits above.
    # Do not treat target=0 as missing; it can represent a real OFF target.

    # Return the cleaned data and a table of rejected values.
    return clean, pd.DataFrame(rejected)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temperature error, position velocity and rolling standard deviation."""
    out = df.copy()

    # Build time axis. The raw collector records laptop time only.
    out["elapsed_seconds"] = parse_clock_times(out["laptop_time"].tolist())

    # Remove duplicate or non-increasing timestamps for derivative calculations
    # by making invalid time deltas NaN; don't remove the original rows.
    dt = out["elapsed_seconds"].diff()
    dt = dt.where(dt > 0)

    # Temperature errors: current minus target, in degrees C.
    out["extruder1_temp_error_c"] = (
        out["extruder1_temp_current"] - out["extruder1_temp_target"]
    )
    out["bed_temp_error_c"] = (
        out["bed_temp_current"] - out["bed_temp_target"]
    )

    # Rolling standard deviation for hotend and bed temperature and each axis.
    rolling_columns = [
        "extruder1_temp_error_c",
        "bed_temp_error_c",
        "head_x_mm",
        "head_y_mm",
        "head_z_mm",
        "head_x_jerk",
        "head_y_jerk",
        "head_z_jerk",
    ]
    for column in rolling_columns:
        out[f"{column}_rolling_std"] = (
            out[column]
            .rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS)
            .std()
        )

    # Approximate velocity from position differences / elapsed seconds.
    # Units: mm/s. A pause or repeated position naturally gives ~0 velocity.
    for axis in ("x", "y", "z"):
        position_col = f"head_{axis}_mm"
        velocity_col = f"head_{axis}_velocity_mm_s"
        out[velocity_col] = out[position_col].diff() / dt

        # Reject only clearly impossible derived velocities; preserve the row
        # and mark the velocity as missing. Tune after inspecting real samples.
        impossible_velocity = out[velocity_col].abs() > 1000.0
        out.loc[impossible_velocity, velocity_col] = np.nan

        out[f"{velocity_col}_rolling_std"] = (
            out[velocity_col]
            .rolling(window=ROLLING_WINDOW, min_periods=MIN_PERIODS)
            .std()
        )

    # The requested X/Y/Z jerk rolling standard deviations are already created
    # above as head_{axis}_jerk_rolling_std.
    # Keep raw validated settings (jerk, max speed, acceleration) as features.

    # Flags for missing source measurements make data quality explicit.
    source_columns = list(LIMITS.keys())
    out["missing_required_metric_count"] = out[source_columns].isna().sum(axis=1)

    return out



# Define export logic for numeric model inputs and separate metadata.
def export_random_forest_files(features: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Write a numeric-only feature matrix plus row metadata for label joining.

    RandomForest estimators should receive only numeric feature columns. The
    timestamp and source section are kept in a parallel file so labels from
    controlled experiments can be joined back to the correct sample rows.
    """
    # Select identifiers/times that help align rows with experimental labels.
    metadata_columns = [c for c in ("laptop_time", "section", "elapsed_seconds") if c in features]
    # Copy metadata into its own table, not into the model feature matrix.
    metadata = features[metadata_columns].copy()

    # Mark non-feature columns that should not be fed to the Random Forest.
    excluded = {"laptop_time", "section", "elapsed_seconds"}
    # Remove timestamps and source section identifiers from the feature table.
    numeric_features = features.drop(columns=[c for c in excluded if c in features])
    # Keep only numeric columns that a model can consume.
    numeric_features = numeric_features.select_dtypes(include=[np.number]).copy()

    # Replace infinities with missing values. Most sklearn estimators require
    # missing values to be imputed before fitting/predicting.
    # Convert infinite values to missing values for later imputation.
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)

    # Save row identifiers separately without adding a pandas index column.
    metadata.to_csv(RF_METADATA_CSV, index=False)
    # Save the numeric Random Forest input matrix.
    numeric_features.to_csv(RF_FEATURES_CSV, index=False)
    # Return the names of the feature and metadata columns for the summary.
    return numeric_features.columns.tolist(), metadata.columns.tolist()


def process_once() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records, parse_errors = parse_log(INPUT_FILE)
    if not records:
        raise RuntimeError(
            f"No /api/v1/printer snapshots found in {INPUT_FILE}. "
            "Check that the file contains the collector's endpoint/timestamp/JSON format."
        )

    raw = pd.DataFrame([extract_row(record) for record in records]).reset_index(drop=True)
    latest_temp = pd.to_numeric(raw["extruder1_temp_current"].iloc[-1], errors="coerce")

    if pd.isna(latest_temp):
        print("WARNING: Extruder temperature reading is missing!")

    # elif latest_temp > MAX_EXTRUDER_TEMP_C:
    #     print("CRITICAL: Extruder temperature too high!")
    #     print(f"Measured temperature: {latest_temp:.1f}°C")

    #     stop_printer()
    #     raise SystemExit(1)
    
    cleaned, rejected = apply_thresholds(raw)
    features = add_features(cleaned)

    features.to_csv(OUTPUT_CSV, index=False)
    rejected.to_csv(REJECTED_CSV, index=False)
    rf_columns, metadata_columns = export_random_forest_files(features)

    summary = [
        "Ultimaker S3 API post-processing summary",
        f"Input file: {INPUT_FILE}",
        f"Output features: {OUTPUT_CSV}",
        f"Random Forest numeric matrix: {RF_FEATURES_CSV}",
        f"Random Forest row metadata: {RF_METADATA_CSV}",
        f"Rows parsed (/api/v1/printer): {len(raw)}",
        f"Metric values masked by thresholds: {len(rejected)}",
        f"Parse warnings: {len(parse_errors)}",
        f"Rolling window: {ROLLING_WINDOW} samples",
        "",
        "Parse warnings:",
        *(parse_errors if parse_errors else ["None"]),
        "",
        "Columns in full output:",
        *features.columns.tolist(),
        "",
        "Numeric Random Forest feature columns:",
        *rf_columns,
        "",
        "Metadata columns kept separately:",
        *metadata_columns,
    ]
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary[:8]))
    print(f"\nSaved features to: {OUTPUT_CSV}")
    print(f"Saved rejected values to: {REJECTED_CSV}")
    print(f"Saved summary to: {SUMMARY_FILE}")



def main() -> None:
    print("Live API post-processing started.")
    print("Checking for new printer data every 2 seconds.")

    while True:
        try:
            process_once()

        except (FileNotFoundError, RuntimeError) as exc:
            print(f"Waiting for printer data: {exc}")

        except Exception as exc:
            print(f"Post-processing error: {exc}")

        time.sleep(2)


if __name__ == "__main__":
    main()
