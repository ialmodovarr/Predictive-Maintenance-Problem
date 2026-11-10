# DATA_PIPELINE.md

## 1. Purpose and Scope

This document specifies the **live data collection and feature-engineering pipeline for Random Forest (RF) and XGBoost** in the Ultimaker S3 predictive-maintenance capstone.

**Current implementation scope:** collect and preserve API and STM32 sensor measurements; synchronize the streams; compute rolling-window, tabular features; and continuously write **two separate feature CSVs** suitable for RF/XGBoost model preparation.

**Future scope, not to implement now:** a 1D-CNN may use raw accelerometer time sequences directly, and may use raw or engineered signals from other sources. Its exact input preparation, resampling, and fusion architecture will be specified separately. **Do not assume CNN inputs must pass through the RF/XGBoost feature engineering pipeline.**

**Modularity is a requirement, not an optional enhancement.** Raw collection, timestamp normalization, time-window selection, feature extraction, storage, and model-specific input preparation must be separate reusable components. The future CNN must be able to reuse the raw measurements and shared timing infrastructure without rewriting the collectors or being constrained to tabular features.

## 2. End-to-End Live Behavior

One master entry-point script must:

1. Prompt the user to designate the run as healthy or failing, and if failing, select one or more affected components.
2. Show the three resulting binary component labels and require confirmation **before** recording.
3. Create a new, uniquely numbered per-print run directory, without overwriting previous runs.
4. Start the Ultimaker API collector and STM32 sensor collector concurrently.
5. Persist timestamped measurements as they arrive, in two separate raw CSVs.
6. Use a shared time reference and window scheduler to define 10-second windows with 2-second strides for **RF/XGBoost feature extraction**.
7. Independently engineer API features and sensor features for each matching window.
8. Append results live to two separate feature CSVs, each with the three print-level labels and matching run/window IDs.
9. Show useful runtime status and errors while recording.
10. On manual stop (e.g., Ctrl+C), gracefully stop collection, flush writes, finish eligible complete windows, and close resources.

Collection and feature computation must occur **during the print**, rather than as a post-print batch. A 2-second stride is a scheduling target; record actual processing delay and report backlogs rather than silently pretending timing is exact.

The master script orchestrates separate components. It must not embed all collection, synchronization, feature engineering, and storage logic in a single large routine.

## 3. Directory Structure and Naming

Use lowercase `snake_case` for directories, files, column names, and feature names.

```text
project/
├── run_pipeline.py                  # Single master entry point
├── pipeline/
│   ├── collectors/
│   │   ├── api_collector.py         # Live Ultimaker API collection
│   │   └── sensor_collector.py      # Live STM32 collection
│   ├── synchronization/
│   │   └── timestamp_sync.py       # Common clock mapping
│   ├── windowing/
│   │   └── rolling_window.py       # Shared window coordinator
│   ├── features/
│   │   ├── api_features.py         # RF/XGBoost API feature extraction
│   │   └── sensor_features.py      # RF/XGBoost sensor feature extraction
│   └── labels/
│       └── label_manager.py        # Print-level labels
├── data/
│   └── runs/
│       ├── run_0001/
│       │   ├── raw/
│       │   │   ├── api_raw.csv
│       │   │   └── sensor_raw.csv
│       │   ├── features/
│       │   │   ├── api_features.csv
│       │   │   └── sensor_features.csv
│       │   └── labels/
│       │       └── print_labels.csv
│       └── run_0002/
│           └── ...
├── models/
│   ├── random_forest/
│   ├── xgboost/
│   └── cnn_1d/                     # Future; out of scope
├── AGENTS.md
├── DATA_PIPELINE.md
├── ML_MODELS.md
└── PROJECT_ARCHITECTURE.md
```

- `api_raw.csv`: timestamped original API observations; retain source endpoint and important raw values, including fields not currently used by RF/XGBoost.
- `sensor_raw.csv`: timestamped STM32 observations from the accelerometer, filament motion sensor, and thermocouple. Preserve native device timestamps, sample indices, original resolution, and relevant raw measurements for **future 1D-CNN usage**.
- `api_features.csv`: continuously appended **RF/XGBoost API engineered features**, one row per scheduled window.
- `sensor_features.csv`: continuously appended **RF/XGBoost sensor engineered features**, one row per scheduled window.
- `print_labels.csv`: authoritative labels for the entire print run.

### Entry point and module contract

Run the entire live RF/XGBoost collection-and-feature pipeline from the project root using:

```bash
python run_pipeline.py
```

`run_pipeline.py` must be the **only command the operator needs to start** for a normal recording. It invokes the startup prompts, initializes the modules, and supervises them until manually stopped. It must not require opening separate terminals for the API collector, sensor collector, or feature preprocessors.

The paths under `pipeline/` above are the **target logical module organization**. Codex should reuse/refactor existing working modules where practical rather than duplicate collectors. It may introduce supporting modules (configuration, writers, tests, package initialization) as needed, but the responsibilities and stable external contracts must remain clear. Do not relocate existing working code purely for aesthetics if equivalent modular structure can be achieved safely; document any deviations.

All paths and runtime outputs should resolve relative to the project configuration/root, not depend on which directory a subprocess happens to be launched from.

Each print must have a unique `run_id` (`run_0001`, `run_0002`, etc.). Never append a new print's data into another run's files. Both raw and feature records must be attributable to their `run_id`.

File naming describes source and representation; **do not name these files as generic CNN-ready training inputs**. A future CNN pipeline may introduce additional derived files/directories without changing the existing raw file contracts.

## 4. Pre-Run Labeling: Binary, Print-Level Ground Truth

Before collection, prompt:

```text
Is this print healthy or failing?
1. Healthy
2. Failing
```

- If healthy, `nozzle_health`, `feeder_health`, and `motion_system_health` are all `healthy`.
- If failing, allow independent selection of **one or more** affected components: nozzle, feeder, motion system. Selected components are `failing`; unselected components are `healthy`.
- Reject a failing selection with zero components selected.
- Show the final three labels and get confirmation before starting.

The only permitted component-label values are `healthy` and `failing` (not `degrading`). **Every feature window from one print receives exactly the same three labels.** The labels are ground-truth experimental annotations, not live ML predictions and not inferred from sensor thresholds. Record them once in `print_labels.csv`, then replicate exactly into both feature outputs.

Required labels: `nozzle_health`, `feeder_health`, `motion_system_health`. If an `overall_system_health` label is ever needed, derive it as `failing` if any component is failing; do not request it separately or change the three primary targets.

**Scientific limitation:** print-level labels may mark a window failing before failure effects become observable. The dataset therefore describes print-level experimental conditions rather than validated instantaneous failure onset or time-to-failure ground truth.

## 5. Live Raw Collection

### 5.1 Ultimaker API

- Collect live from `/api/v1/print_job` and `/api/v1/printer` at an approximately 1 Hz target rate; use actual timestamps, not assumed spacing.
- Save raw observations during the run, including source endpoint and host UTC collection/receipt timestamp.
- Use reasonable request timeouts, error reporting, and recovery behavior. Collector delays must not freeze the other collector.
- Preserve useful fields for potential future model approaches even if current RF/XGBoost feature extraction uses only a subset.

### 5.2 STM32 sensors

- Accelerometer X/Y/Z
- Filament motion sensor
- Thermocouple
- Preserve each observation's meaningful sensor and native timestamp information. Do **not** downsample raw accelerometer readings to a lower frequency rate or replace sequences with aggregates in raw storage.
- Different sensor frequencies do not require artificial shared sample rows. Missing columns in an event-style/raw row are not zero readings.
- Store raw readings at the highest available fidelity required to reconstruct accelerometer sequences for future CNN experiments.

## 6. Shared Timestamps and Synchronization

- Use host-computer **UTC** as the cross-source reference; standardized timestamp names are `timestamp_utc`, `window_start_utc`, and `window_end_utc`.
- Use ISO 8601 UTC timestamps with milliseconds or finer precision, e.g. `2026-11-09T21:00:00.000Z`.
- Preserve STM32 native microsecond/device timestamps separately. Implement a documented mapping from device clock time to host UTC, including offset estimation; handle resets, serial latency, and drift as far as practical. Record or flag uncertain synchronization rather than presenting inaccurate precision.
- The API and sensors must not each independently define their own zero-time and assume those elapsed times correspond.
- Use shared run start/anchor and one window schedule for both RF/XGBoost feature extractors.
- Synchronization and timestamp normalization must be independent of the RF/XGBoost feature definitions so a future CNN pipeline can reuse them.

## 7. RF/XGBoost Windowing (Not a CNN Constraint)

The **current tabular RF/XGBoost pipeline** uses:

```text
window_duration_seconds = 10
window_stride_seconds = 2
```

Examples of windows relative to the shared anchor: `[0,10)`, `[2,12)`, `[4,14)`, `[6,16)`. Windows overlap by 8 seconds and are half-open, so a sample with timestamp equal to the end boundary is excluded from that window.

- After the first 10-second interval completes (plus any explicitly configured small late-arrival allowance), engineer the first window; process the next window every 2 seconds.
- The shared coordinator issues the **same window IDs and UTC boundaries** for API and sensor processors.
- Use timestamp ranges, never a fixed number of API samples, to select data. The actual number of samples can vary.
- Do not process the same `run_id` + `window_id` twice. Allow for temporary processing lag and report it.
- Retain raw measurements independently of the window processor so another model can use a different window duration, stride, sample rate, or raw sequence representation without recollection.

## 8. RF/XGBoost Feature Engineering

Each extractor receives measurements belonging to an assigned shared window and returns a feature record plus data-quality metadata. Keep source-specific feature functions distinct, and make the feature specification easy to change without modifying collectors or timestamp synchronization.

### 8.1 API feature set

Extruder 1 (4 features):
- `api_extruder_temperature_mean` — average measured temperature within the 10-second window.
- `api_extruder_temperature_std` — rolling/window standard deviation of measured temperature within the same 10-second window.
- `api_extruder_temperature_slope` — measured temperature change over time, using actual timestamps.
- `api_extruder_temperature_error` — mean of `(current_temperature - target_temperature)` across valid paired observations in the 10-second window. Keep the sign: positive means measured temperature exceeds target. When target measurements are unavailable, mark this feature missing rather than assuming a value.

Print head (2 features), derived from X/Y/Z position values and **actual elapsed timestamps**:
- `api_head_velocity_mean`
- `api_head_velocity_std`

By default, velocity here means scalar 3D speed derived from consecutive valid positions and real elapsed time; document units and formula.

**Feature count:** API = **6** engineered features (4 extruder + 2 print-head).

**Experimental-design warning (retain the feature anyway):** `api_extruder_temperature_error` is intentionally included as requested. Because target-temperature changes may be used to simulate nozzle faults, this feature can encode the experimental intervention rather than physical nozzle degradation. Document this risk, and consider feature-ablation evaluation (with vs. without temperature error) when interpreting model performance. Do **not** remove the feature from the live output.

### 8.2 External sensor feature set

Accelerometer (15 features: five metrics for each of X/Y/Z):
- `accel_x_rms`, `accel_y_rms`, `accel_z_rms`
- `accel_x_std`, `accel_y_std`, `accel_z_std`
- `accel_x_peak_to_peak`, `accel_y_peak_to_peak`, `accel_z_peak_to_peak`
- `accel_x_dominant_frequency`, `accel_y_dominant_frequency`, `accel_z_dominant_frequency`
- `accel_x_kurtosis`, `accel_y_kurtosis`, `accel_z_kurtosis`

Filament motion (4 features):
- `filament_velocity_mean`
- `filament_movement_rate_std`
- `filament_movement_fraction_pct` � percentage of the scheduled window occupied by valid intervals with a change in encoder position, including retractions.
- `filament_longest_no_movement_interval_s` � longest continuous observed duration with unchanged encoder position, in seconds. Gaps and clock-segment boundaries break continuity.

Thermocouple (3 features):
- `thermocouple_temperature_mean`
- `thermocouple_temperature_std`
- `thermocouple_temperature_slope`


**Feature count:** external sensors = **22** engineered features (15 accelerometer + 4 filament + 3 thermocouple). Together with the six API features, the RF/XGBoost pipeline must output **28 engineered features** per matched, sufficiently populated window: **6 in `api_features.csv`** and **22 in `sensor_features.csv`**. Identifiers, quality metadata, and the three health labels do not count toward the 28 model input features.

Document physical units, feature formulas, minimum sample requirements, and behavior for irregular samples. These engineered features are **RF/XGBoost tabular inputs**, not a mandatory representation for the future 1D-CNN.

## 9. Live Feature Outputs and Schema

Generate and incrementally append:

- `features/api_features.csv`
- `features/sensor_features.csv`

Both feature files must contain:

```text
run_id
window_id
window_start_utc
window_end_utc
nozzle_health
feeder_health
motion_system_health
```

Use stable, unique `window_id` values such as `window_0001`; corresponding API/sensor rows must have the same `run_id`, `window_id`, start, and end. Include source-appropriate sample counts, coverage/timing diagnostics, and missing-feature reasons. Do not assign an overall quality classification to a window.

- Each source produces one traceable result for each scheduled window even when its measurements are insufficient, so later joins can identify gaps.
- Record valid available features; leave unsupported features missing. Never silently replace missing values with zero.
- Append each completed window once, rather than repeatedly recomputing and rewriting all past output.
- The feature outputs are aligned but **remain separate files**. No permanently saved combined feature CSV is required.
- Labels must be identical in both files for the same run, and must agree with `print_labels.csv`.
- Do not include these labels as input features during training.

## 10. Missing Data and Reliability

A window may be incomplete due to missing API polls, sparse sensor samples, serial interruptions, invalid readings, or synchronization uncertainty.

- Preserve its identifiers and times.
- Compute features only when enough valid data exists; otherwise leave affected features missing.
- Record sample counts, observed coverage, gaps, synchronization issues, and feature-specific missing reasons. Do not classify or reject whole windows during live processing; window selection and missing-value handling belong to later model preparation. Features still require valid inputs and timestamps before calculation.
- Make minimum coverage/sample thresholds configurable and document them separately for API, accelerometer, filament, and thermocouple; do not assume identical requirements for all sources.
- Preserve raw data for later review and alternate reprocessing.
- Track collection/processing lag, prevent duplicate rows, and report errors.
- Use incremental processing and bounded working buffers rather than rereading the entire recording each cycle.
- Handle device reset and API disconnection explicitly. Do not fabricate missing elapsed time or silently merge separate runs.

## 11. Modular Architecture and Future 1D-CNN Extensibility

**Critical scope boundary: this document's feature-engineering and feature-output requirements are for RF and XGBoost only.**

Organize implementation into independently testable responsibilities:

1. **Run controller / CLI** — label selection, run creation, start/stop, process supervision, status.
2. **API collector** — acquire and persist timestamped raw API records.
3. **STM32 sensor collector** — acquire and persist full-fidelity timestamped raw sensor records.
4. **Time synchronization** — map native device timestamps to common UTC and expose validated times.
5. **Window scheduler / selector** — reusable timestamp-based time selection; default 10s/2s configuration applies to RF/XGBoost.
6. **API tabular feature extractor** — API-window feature computation for RF/XGBoost.
7. **Sensor tabular feature extractor** — sensor-window feature computation for RF/XGBoost.
8. **Feature writer / schema validation** — incremental, consistent separate CSV outputs with labels and quality metadata.
9. **Model-input preparation** — later combine matching tabular windows for RF/XGBoost, without coupling collection to a particular model.

**Future CNN requirements (compatibility only, not implementation now):**

- The raw accelerometer X/Y/Z time series and original timestamps must remain available at full useful resolution.
- The future CNN may operate directly on raw sequences, rather than the 15 engineered accelerometer statistics in `sensor_features.csv`.
- It may use different segmentation, resampling, tensor layouts, and hybrid fusion with lower-frequency raw or engineered inputs; these choices are not finalized here.
- A future CNN data-preparation component must be able to consume the existing raw streams and synchronization metadata without modifying the live collectors.
- Existing RF/XGBoost feature extraction must not be hardwired into the only possible downstream data path.
- **Do not build, train, or assume a particular CNN architecture as part of implementing this specification.** See `ML_MODELS.md` for model-level specifications when those are finalized.

## 12. RF/XGBoost Model Preparation and Leakage Risks

- Join the two aligned feature outputs for a run using `run_id` and `window_id`; verify matching window UTC boundaries.
- Form tabular model inputs from engineered columns only. Keep the three component health columns as target labels.
- Prepare data for RF/XGBoost independently of live collection; model training itself need not occur while collecting samples.
- Split training, validation, and test **by independent print run**, never random overlapping windows, to avoid leakage from nearly identical time intervals.
- Remember that print-level labels describe the controlled experimental condition, not confirmed instantaneous onset of failure.
- Retain API temperature error as a required feature, but report its possible leakage from experimentally manipulated temperature targets; evaluate a model without that feature as a sensitivity check when feasible.

## 13. Verification and Acceptance Criteria

Codex should inspect and reuse functioning existing code when practical, change incompatible behavior, and add synthetic-data tests. Completion requires all of the following:

### Control and labels
- [ ] One master entry point starts both collectors and live RF/XGBoost preprocessing.
- [ ] User selects healthy or failing before recording; failing requires at least one component.
- [ ] Confirmation is required; labels remain identical for every window of a run.
- [ ] Unique per-run folders and all five named CSV files are produced safely.
- [ ] Manual stop gracefully flushes and closes resources.

### Data and timing
- [ ] API and STM32 raw streams are written live and retain appropriate timestamps.
- [ ] Native STM32 timestamps and full-resolution raw accelerometer sequences are preserved.
- [ ] API-to-STM32 timing alignment is explicitly implemented or uncertainties are flagged.
- [ ] RF/XGBoost windows are 10 seconds long, with 2-second stride and identical boundaries across sources.
- [ ] Window IDs, timestamps, and labels match between both feature CSVs.
- [ ] Late/missing/invalid observations are handled and visibly flagged.
- [ ] No duplicate windows; processing is incremental and lag is monitored.

### Feature extraction
- [ ] All specified API and sensor engineered features are computed correctly when inputs exist.
- [ ] `api_extruder_temperature_error` is included and correctly calculated from current minus target temperature, using paired values within each window.
- [ ] The feature schema contains exactly 6 API features and 22 sensor features (28 engineered features total), excluding metadata and health labels.
- [ ] Features are saved live in **two separate append-only feature datasets**, not a merged file.
- [ ] Labels are excluded from model predictor columns.

### Modularity and future extensibility
- [ ] Collectors do not depend on RF/XGBoost feature extraction.
- [ ] Clock synchronization and raw retention are reusable independently of the tabular feature path.
- [ ] Window parameters and feature specifications are configurable rather than hardcoded throughout the codebase.
- [ ] It will be possible to add a future raw-sequence 1D-CNN preparation branch without altering existing collectors or forcing the CNN through RF/XGBoost feature CSVs.
- [ ] **No CNN-specific implementation is required or implied by this document.**

### Tests
- [ ] Synthetic streams with differing frequencies verify synchronization, window boundaries, and matching IDs.
- [ ] Tests cover missing data, out-of-order/late samples, repeated polling, timestamp resets, and graceful shutdown.
- [ ] Tests confirm label persistence and consistency across both outputs.
- [ ] A hardware integration test is performed separately; code review alone is not evidence of live timing performance.

## 14. Implementation Principles for Codex

- Treat this file as the authoritative **RF/XGBoost live pipeline** specification, not a universal data-processing design for all models.
- Preserve working hardware communication and raw data fidelity when possible.
- Prefer small, independently testable modules with explicit input/output contracts.
- Do not introduce assumptions or transformations that destroy information needed by future models.
- Document deviations, especially clock mapping, sample coverage thresholds, and feature computation definitions.
- Keep model training and future CNN architecture outside this implementation task unless separately requested.

## Software implementation

The recorder is available through `python run_pipeline.py`. See [operator instructions](README.md) and [runtime architecture](PROJECT_ARCHITECTURE.MD) for settings, storage, timing and hardware validation.

### Timing recovery implementation

Forward device-clock discontinuities, resets and sequence gaps are explicitly
diagnosed. Ambiguous preceding segments are excluded from pending feature buffers,
with counts retained in diagnostics. Raw observations remain available. Filament
velocity never bridges segmented gaps. Unknown-rate timing limits default to the
configurable sensor_gap_s value. See PROJECT_ARCHITECTURE.MD for the fixed-offset
UTC mapping and its remaining latency/drift uncertainty.
