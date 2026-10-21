"""
Post-process Ultimaker S3 API logs for predictive-maintenance features.
Expected project layout:
data/
    process_api_data.py        <- this script
    api_data/
        api_data_YYYY-MM-DD_HH-MM-SS.txt <- timestamped raw logs written by the API collector
    processed/
        printer_features_<run timestamp>.csv
        random_forest_features_<run timestamp>.csv
        rejected_values_<run timestamp>.csv
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
from datetime import datetime
from pathlib import Path
from typing import Any
import time
import numpy as np
import pandas as pd
PRINTER = "http://172.27.135.1"
USERNAME = "6bf926c99944ad9e15da49c2ed1bffb5"
PASSWORD = "716b548893f2fe16f0d1929b94857484d291b8af67e61685ba9c1ad24f682e30"
# ---------------------------------------------------------------------------
# Paths: this script is in data/, and raw timestamped logs are in data/api_data/.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DIR = SCRIPT_DIR / "api_data"
OUTPUT_DIR = SCRIPT_DIR / "processed"

# One unique set of output files is created each time this script starts.
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
OUTPUT_CSV = OUTPUT_DIR / f"printer_features_{RUN_TIMESTAMP}.csv"
RF_FEATURES_CSV = OUTPUT_DIR / f"random_forest_features_{RUN_TIMESTAMP}.csv"
REJECTED_CSV = OUTPUT_DIR / f"rejected_values_{RUN_TIMESTAMP}.csv"

POLL_INTERVAL_SECONDS = 2

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
    """Extract complete /api/v1/printer JSON snapshots from one collector log.

    The collector may still be writing the final snapshot. Incomplete JSON is
    reported as a parse warning and will be retried on the next polling cycle.
    """
    if not path.exists():
        return [], []

    text = path.read_text(encoding="utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    errors: list[str] = []

    # Bound each endpoint section by the next endpoint header. This avoids
    # relying on exact divider-line formatting.
    endpoint_matches = list(re.finditer(r"/api/v1/[^\s]+", text))
    for section_number, endpoint_match in enumerate(endpoint_matches, start=1):
        endpoint = endpoint_match.group(0).strip()
        if endpoint != "/api/v1/printer":
            continue

        next_start = (
            endpoint_matches[section_number].start()
            if section_number < len(endpoint_matches)
            else len(text)
        )
        body = text[endpoint_match.end():next_start]
        timestamp_match = re.search(r"Laptop timestamp:\s*([^\r\n]+)", body)
        if not timestamp_match:
            errors.append(f"{path.name}, section {section_number}: missing laptop timestamp")
            continue

        json_start = body.find("{", timestamp_match.end())
        if json_start < 0:
            errors.append(f"{path.name}, section {section_number}: missing JSON payload")
            continue

        try:
            payload, _ = json.JSONDecoder().raw_decode(body[json_start:].lstrip())
        except json.JSONDecodeError as exc:
            errors.append(f"{path.name}, section {section_number}: incomplete/invalid JSON ({exc})")
            continue

        records.append({
            "laptop_time": timestamp_match.group(1).strip(),
            "payload": payload,
            "section": section_number,
            "source_file": path.name,
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
        "source_file": record.get("source_file", ""),
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
# Export a numeric-only feature matrix for Random Forest.
def export_random_forest_files(features: pd.DataFrame) -> list[str]:
    """Write only numeric model inputs; timestamps remain in printer_features."""
    excluded = {"laptop_time", "section", "elapsed_seconds", "source_file"}
    numeric_features = features.drop(
        columns=[column for column in excluded if column in features]
    )
    numeric_features = numeric_features.select_dtypes(include=[np.number]).copy()
    numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
    numeric_features.to_csv(RF_FEATURES_CSV, index=False)
    return numeric_features.columns.tolist()


def discover_api_files() -> list[Path]:
    """Return collector log files in chronological order."""
    if not INPUT_DIR.exists():
        return []
    files = [
        path for path in INPUT_DIR.glob("*.txt")
        if path.is_file() and path.name.startswith("api_data")
    ]
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create this run's three output files immediately, then refresh them as
    # additional snapshots arrive.
    for output_path in (OUTPUT_CSV, RF_FEATURES_CSV, REJECTED_CSV):
        output_path.touch(exist_ok=False)

    existing_files = discover_api_files()
    if not existing_files:
        print(f"Waiting for API logs in: {INPUT_DIR}")
        initial_file = None
    else:
        # Start with the most recently modified raw API log, not older sessions.
        initial_file = existing_files[-1]
        print(f"Starting from newest API log: {initial_file.name}")

    # In-memory state only: no metadata/progress file is created.
    watched_files: list[Path] = [initial_file] if initial_file else []
    baseline_mtime = initial_file.stat().st_mtime if initial_file else None
    seen_keys: set[tuple[str, str, int]] = set()
    all_rows: list[dict[str, Any]] = []

    print("Live API post-processing started.")
    print(f"Watching {INPUT_DIR} for new api_data*.txt files every {POLL_INTERVAL_SECONDS}s.")
    print("Output CSVs for this run:")
    print(f"  Printer features: {OUTPUT_CSV}")
    print(f"  Random Forest features: {RF_FEATURES_CSV}")
    print(f"  Rejected values: {REJECTED_CSV}")
    print("Press Ctrl+C to stop.")

    while True:
        try:
            available_files = discover_api_files()

            # Keep the initial newest file, then add files created/rotated later.
            # Existing older logs are intentionally not loaded.
            if initial_file is None and available_files:
                # If logs appear after startup, begin with the newest one available.
                initial_file = available_files[-1]
                baseline_mtime = initial_file.stat().st_mtime
                watched_files.append(initial_file)
                print(f"Now processing newest API log: {initial_file.name}")

            for path in available_files:
                if path in watched_files:
                    continue
                if baseline_mtime is not None and path.stat().st_mtime >= baseline_mtime:
                    watched_files.append(path)
                    print(f"Detected API log: {path.name}")

            # Read watched logs again so newly appended snapshots are picked up.
            for path in watched_files:
                records, errors = parse_log(path)
                if errors:
                    for warning in errors:
                        print(f"Parse warning: {warning}")
                for record in records:
                    key = (record["source_file"], record["laptop_time"], record["section"])
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)
                    all_rows.append(extract_row(record))

            if all_rows:
                raw = pd.DataFrame(all_rows).reset_index(drop=True)
                latest_temp = pd.to_numeric(
                    raw["extruder1_temp_current"].iloc[-1], errors="coerce"
                )
                if pd.isna(latest_temp):
                    print("WARNING: Latest extruder temperature reading is missing!")

                cleaned, rejected = apply_thresholds(raw)
                features = add_features(cleaned)

                features.to_csv(OUTPUT_CSV, index=False)
                rejected.to_csv(REJECTED_CSV, index=False)
                rf_columns = export_random_forest_files(features)

                print(
                    f"Updated {len(features)} printer rows; "
                    f"{len(rejected)} rejected measurements; "
                    f"{len(rf_columns)} Random Forest features."
                )
            else:
                print("No complete /api/v1/printer snapshots found yet.")

        except KeyboardInterrupt:
            print("\\nPost-processing stopped.")
            break
        except Exception as exc:
            print(f"Post-processing error: {exc}")

        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
