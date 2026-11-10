# Legacy pipeline scripts

These scripts are preserved byte-for-byte from their original locations. They are
reference implementations for the modular pipeline specified in DATA_PIPELINE.md.
The live recorder is available from the repository root via run_pipeline.py.

| Original path | Preserved path | Reuse destination / role |
| --- | --- | --- |
| data/api.py | legacy/data/api.py | pipeline/collectors/api_collector.py |
| data/sensors/print_logger.py | legacy/data/sensors/print_logger.py | pipeline/collectors/sensor_collector.py; standalone Excel reporting reference |
| data/preprocessed_api.py | legacy/data/preprocessed_api.py | pipeline/features/api_features.py; parsing and validation reference |
| data/sensor_preprocessing.py | legacy/data/sensor_preprocessing.py | pipeline/features/sensor_features.py; numerical helper reference |
| data/debugging_preprocessing.py | legacy/data/debugging_preprocessing.py | Alternate API debugging tool |
| data/accelerometer_data.py | legacy/data/accelerometer_data.py | Older standalone accelerometer preprocessing tool |

## Running archived scripts

Moving these files changes their execution context. API scripts resolve some paths
relative to their script location; sensor scripts use the working directory for
some paths. They are not drop-in launch commands for the original data folders
and may require explicit path adjustments before reuse. Review output destinations
before running them. No automatic legacy launch compatibility is provided.

## Preserved project assets

Existing raw logs, CSVs, Excel workbooks, requirements, model code, and research
notes remain at their original paths. Historical recordings have not been assigned
new run IDs, inferred labels, or inferred synchronization. sensortest.ino remains a
standalone hardware test at the repository root. The existing Random Forest code
still uses the old labeling and alignment workflow and needs separate adaptation.

New recordings will belong under data/runs/ when the live recorder is used.
Do not treat these archived scripts as satisfying the new acceptance criteria.
