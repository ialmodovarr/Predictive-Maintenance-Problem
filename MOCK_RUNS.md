# Mock printer recordings

30 runs span 9.65 hours with 63,709 raw API records and 26,609,816 sensor packets. There are 17,246 paired windows, including 16,941 with all 28 features. Run files occupy 14.50 GB.

Eight hybrid runs preserve original data and 22 are fully synthetic. Fifteen have healthy nozzles and fifteen have failing nozzles. Feeder and motion labels are healthy throughout.

|Run|Seconds|Nozzle|Feeder|Motion|Origin|API|Sensor|Windows|Split|
|---|---:|---|---|---|---|---:|---:|---:|---|
|run_0001|1423.0|healthy|healthy|healthy|hybrid|1789|1091413|707|train_validation|
|run_0002|1303.5|failing|healthy|healthy|hybrid|1622|999780|647|train_validation|
|run_0003|1042.0|failing|healthy|healthy|hybrid|1331|799174|516|test|
|run_0004|968.8|failing|healthy|healthy|hybrid|1223|743062|480|train_validation|
|run_0005|851.5|failing|healthy|healthy|hybrid|1054|653075|421|train_validation|
|run_0006|1085.4|failing|healthy|healthy|hybrid|1260|818364|538|train_validation|
|run_0007|1200.0|healthy|healthy|healthy|hybrid|2388|919362|596|test|
|run_0008|1167.0|healthy|healthy|healthy|synthetic|2334|895065|579|train_validation|
|run_0009|950.2|failing|healthy|healthy|hybrid|1184|698579|471|test|
|run_0010|1309.0|healthy|healthy|healthy|synthetic|2618|1003975|650|train_validation|
|run_0011|1013.0|healthy|healthy|healthy|synthetic|2026|776951|502|train_validation|
|run_0012|1194.0|healthy|healthy|healthy|synthetic|2388|915774|593|test|
|run_0013|1316.0|healthy|healthy|healthy|synthetic|2632|1009344|654|train_validation|
|run_0014|1338.0|healthy|healthy|healthy|synthetic|2676|1026218|665|train_validation|
|run_0015|959.0|healthy|healthy|healthy|synthetic|1918|735533|475|train_validation|
|run_0016|969.0|healthy|healthy|healthy|synthetic|1938|743203|480|train_validation|
|run_0017|1129.0|healthy|healthy|healthy|synthetic|2258|865919|560|train_validation|
|run_0018|1220.0|healthy|healthy|healthy|synthetic|2440|935716|606|train_validation|
|run_0019|1241.0|healthy|healthy|healthy|synthetic|2482|951823|616|test|
|run_0020|998.0|healthy|healthy|healthy|synthetic|1996|765446|495|train_validation|
|run_0021|1167.0|healthy|healthy|healthy|synthetic|2334|895065|579|train_validation|
|run_0022|1121.0|failing|healthy|healthy|synthetic|2242|859783|556|train_validation|
|run_0023|969.0|failing|healthy|healthy|synthetic|1938|743203|480|train_validation|
|run_0024|1480.0|failing|healthy|healthy|synthetic|2960|1135128|736|train_validation|
|run_0025|1232.0|failing|healthy|healthy|synthetic|2464|944920|612|train_validation|
|run_0026|1328.0|failing|healthy|healthy|synthetic|2656|1018548|660|train_validation|
|run_0027|1309.0|failing|healthy|healthy|synthetic|2618|1003975|650|test|
|run_0028|952.0|failing|healthy|healthy|synthetic|1904|730164|472|train_validation|
|run_0029|1495.0|failing|healthy|healthy|synthetic|2990|1146633|743|train_validation|
|run_0030|1023.0|failing|healthy|healthy|synthetic|2046|784621|507|train_validation|

## Structure and reproduction

Generate a fresh dataset with `python -m pipeline.mock_build`. For selected missing runs use `python -m pipeline.mock_dataset 8 22`. Move existing mock outputs and their metadata aside before rebuilding. Use Python 3.11+ and the package versions in data/mock_metadata/reference.json. Per-run seeds combine the config.json seed and run number. Run `python -m pipeline.mock_audit` for the final audit.

Each data/mock_runs/run_NNNN directory contains raw/api_raw.csv, raw/sensor_raw.csv, features/api_features.csv, features/sensor_features.csv, labels/print_labels.csv and settings.json. Select explicitly using the shared model loader `load_run(Path("data/mock_runs/run_0008"))`. Model CLIs default to data/runs. Use data/mock_metadata/split.json to select whole runs for development and held-out workflow testing.

## Generation and supplementation

Each synthetic run shares its head-motion and thermal trajectory across API, vibration, filament and lagged thermocouple measurements. Normal pauses and intermittent retractions occur in both conditions. Run-specific oscillations, quantized counts, timing jitter and held sensor values generate raw STM32 packets. The measured historical packet period is approximately 1303 microseconds. Nominal feature thresholds remain at 800 Hz. The generator targets 767 Hz, with provisional encoder and temperature updates at 10 Hz and 3 Hz. Synthetic thermal fluctuations and intermittent feed reductions vary with provisional nozzle severity. The target temperature distribution is shared across classes. Historical interventions that changed temperature targets remain in real records.

Original CSV rows are retained in order. Hybrid sensor additions follow recorded head position and temperature. Metadata records their UTC intervals. For run 0007, API observations after the original API row count and sensor packets after the original segment extend recording to 20 minutes. Null thermocouple fields in original packets remain missing. Historical replay metadata for 0004 and 0005 identifies incorrect sessions. The generator uses source records and actual feature anchors instead.

Features use the existing scheduler and extractors on raw observations with the existing timing diagnostics and validity thresholds. Ten-second half-open windows start every two seconds. ML_MODELS.md is empty, and the CNN input contract remains provisional.

## Pilot and final validation

Early pilot windows had low acceleration variability and nearly constant frequency. A second trial coupled vibration too tightly to API head speed. The final vibration envelope varies with time and operating mode. Rebuilding run 0008 reproduced all six files byte for byte.

|Pilot|Complete windows|Paired windows|Seconds to generate|GB|
|---|---:|---:|---:|---:|
|run_0008|579|579|202.4|0.49|
|run_0022|556|556|196.2|0.47|

The pilots projected 15.01 GB for ten hours, with 145.4 GB free. Generation and feature processing took 2.49 summed worker-hours. validation.json contains hashes, sensor parser checks, feature completeness, means, standard deviations and percentiles by run.

## Comparison with real recordings

Medians over finite windows are shown below. Runs 0007 and 0009 contributed held-out sensor references. Run 0007 supplies the only real thermocouple feature distribution.

|Recording|Head speed (mm/s)|X acceleration SD (m/s²)|X dominant frequency (Hz)|Filament velocity (mm/s)|Thermocouple (°C)|
|---|---:|---:|---:|---:|---:|
|run_0001|6.87|missing|missing|missing|missing|
|run_0006|6.71|1.15|191.02|0.32|missing|
|run_0007|28.33|1.10|134.91|0.11|67.48|
|run_0009|6.93|2.66|61.31|-0.00|missing|
|run_0008|10.19|1.05|190.20|0.27|66.75|
|run_0022|9.00|1.05|62.31|0.24|66.55|

## Limitations

Synthetic and supplemented measurements lack experimental validation. The simulated vibration, thermal and nozzle-fault responses are provisional. Results measure model-development workflow behavior under these assumptions. Independent printer experiments are required to establish real-world accuracy. All feeder and motion labels are healthy, so failure detection for these components requires additional experiments.

A separate rebuild of run 0022 also matched all six file hashes. Its single-process peak working set was 54.4 MiB.
