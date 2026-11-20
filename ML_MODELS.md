# Machine learning models

## Scope and evidence

The hybrid 1D-CNN classifies nozzle health only: healthy = 0, failing = 1.
Existing feeder and motion labels and RF behavior are preserved. Every
window inherits its print label; this is not a verified instantaneous fault-onset
label. Scores are not calibrated physical probabilities. No live CNN integration
is implemented. `run_pipeline.py` remains the live recording/feature entry point.

Mock recordings are temporary workflow data. Generated vibration, thermal and
fault responses are provisional. Shared references include held-out recordings,
so run-isolated mock tests are not independent experimental validation. Never
combine `data/mock_runs` and `data/runs` implicitly. Temperature error can encode
experimental temperature interventions; retain it in this approved baseline and
interpret results accordingly. Stationary filament intervals can be normal pauses.

## Random Forest and XGBoost

Random Forest retains its original real-data default and leave-one-run-out evaluation.
XGBoost now trains nozzle_health only (healthy=0, failing=1), defaulting to
`data/mock_runs`. Select `--dataset runs` explicitly for real recordings and provide
`--split-file` with disjoint train_validation and test run lists. Unexpected labels,
mixed feature versions, mismatched windows and invalid numeric inputs are rejected.
The shared RF per-run loader remains unchanged. All 28 features remain selected by
default; --features and --exclude remain available, including temperature error.
XGBoost validates predictor names against the six API and 22 sensor features in
DATA_PIPELINE.md. Unexpected feature columns are rejected during training and
prediction, including numeric generation metadata. Saved-model feature selections
are checked against the same allowlist; CSV and model feature order is preserved.

XGBoost uses the predefined mock split and the CNN's four run-stratified folds
(sorted development IDs, shuffle=True, seed=42). Saved CNN fold assignments are
checked when available. Each fold trains from scratch; balanced sample weights
are calculated from that fold's training windows. The final model starts fresh on
all development runs, then scores the held-out test runs. Hyperparameters remain
300 trees, depth 3, learning rate .05, subsample .8, colsample_bytree .8,
min_child_weight 5, reg_lambda 1, binary:logistic, logloss, hist, default max_bin
256, n_jobs=-1 and seed 42. Threshold is .5; no early stopping, tuning, calibration,
scaling or imputation is performed. Native NaN handling retains partial windows.
All-selected-features-missing windows receive insufficient_data and blank predictions;
they are counted explicitly and excluded from metric calculations and fitting.

Generated outputs are under models/xgboost/artifacts/<mock_runs|runs>/:
nozzle_health_model.joblib, nozzle_health_oof_predictions.csv,
nozzle_health_test_predictions.csv, nozzle_health_feature_importance.csv and
metrics.json. Existing nonempty output directories refuse training; deliberately
archive prior outputs before rerunning. Legacy outputs remain untouched.
The model bundle retains model/features/target/threshold and adds dataset,
feature_version, training_runs, creation time and library versions. Only the final
model is saved. metrics.json retains the existing evaluation path for pooled OOF
metrics and adds fold_evaluations, test_evaluation, exact splits, effective model
configuration, separate coverage diagnostics and eligibility counts. Precision,
recall, F1, balanced accuracy, confusion matrices and per-run summaries retain
Bella's calculations; coverage warnings retain the >=10 percentage-point rule.

Prediction loads only the selected dataset's nozzle model, checks feature version
and ordered inputs, and accepts fully unlabeled feature files. If labels are present,
the existing label consistency checks apply. Outputs under predictions/<dataset>/
retain identifiers, UTC boundaries, failing score and predicted class, adding status,
available_feature_count and seen_in_training. A run absent from training is not
necessarily independent of synthetic generation. Scores are not calibrated failure
probabilities. Prediction remains offline; no live system is implemented.

```powershell
python models/xgboost/xgboost_model.py train --dataset mock_runs
python models/xgboost/xgboost_model.py train --dataset runs --split-file path/to/split.json
python models/xgboost/xgboost_model.py predict --dataset mock_runs --run run_0008
python models/xgboost/xgboost_model.py predict --dataset runs --run run_0009
```

Mock tests retain shared generation references, including held-out recordings;
these are workflow evaluations, not independent experimental validation. CNN and
XGBoost share run partitions but differ in eligible windows. RF's existing outputs
use a different dataset and protocol. Direct performance comparisons require these
differences to be addressed explicitly.

## Approved hybrid architecture

PyTorch; joint end-to-end optimization. Inputs: acceleration `(B,3,7670)` and
standardized auxiliary features `(B,13)` at the provisional mock rate of 767 Hz.

| Stage | Settings | Output excluding batch | Parameters |
|---|---|---|---:|
| Conv1/ReLU | 3 to 16, kernel 15, stride 1, padding 7 | 16 x 7670 | 736 |
| MaxPool1 | kernel/stride 2, no padding, floor mode | 16 x 3835 | 0 |
| Conv2/ReLU | 16 to 32, kernel 7, stride 1, padding 3 | 32 x 3835 | 3616 |
| MaxPool2 | kernel/stride 2 | 32 x 1917 | 0 |
| Conv3/ReLU | 32 to 64, kernel 5, stride 1, padding 2 | 64 x 1917 | 10304 |
| MaxPool3 | kernel/stride 2 | 64 x 958 | 0 |
| Adaptive average pooling, flatten | temporal length 1 | 64 | 0 |
| Concatenate D and T | 64 learned plus 13 engineered | 77 | 0 |
| Linear/ReLU/Dropout | 77 to 32, dropout 0.2 | 32 | 2496 |
| Linear output | 32 to 1 | one logit | 33 |

Total: **17,185 trainable parameters**, with biases. Dilation is 1. No dense
auxiliary branch, attention, recurrent layer, batch normalization, convolutional
dropout, or raw-sequence flattening. Kaiming normal initialization applies to
convolutions and hidden linear weights; their biases are zero. The output retains
PyTorch initialization. The local receptive field is 50 samples (about 65 ms);
global pooling summarizes local responses over the whole window, losing event
ordering. Max pooling is not an anti-aliasing filter.

Eren (2017), DOI 10.1155/2017/8617315, provides precedent for compact raw-vibration
CNNs. This fusion model is not a reproduction, and bearing results do not establish
nozzle performance. Layer settings are an engineering baseline, not proven optimal.

## Auxiliary feature order and units

1. `api_extruder_temperature_mean` (degrees C)
2. `api_extruder_temperature_std` (degrees C)
3. `api_extruder_temperature_slope` (degrees C/s)
4. `api_extruder_temperature_error` (current minus target, degrees C)
5. `api_head_velocity_mean` (scalar 3D speed, mm/s)
6. `api_head_velocity_std` (mm/s)
7. `filament_velocity_mean` (signed mm/s, including retractions)
8. `filament_movement_rate_std` (mm/s)
9. `filament_movement_fraction_pct` (percentage, not fraction)
10. `filament_longest_no_movement_interval_s` (seconds)
11. `thermocouple_temperature_mean` (degrees C)
12. `thermocouple_temperature_std` (degrees C)
13. `thermocouple_temperature_slope` (degrees C/s)

Read existing saved features without recomputing them. All 13 must be finite.
Do not impute, forward-fill, add missingness indicators, or require the 15 unused
engineered acceleration features to be present. See FEATURE_DEFINITIONS_PROPOSAL.md
for formulas and DATA_PIPELINE.md for the complete hybrid input contract.

## Window preparation and cache

Use existing IDs and exact UTC boundaries: ten-second half-open windows every two
seconds. Match API and sensor outputs by run/window ID and verify timestamps and
labels. The raw branch uses mapped UTC; receipt time is not an acquisition clock.
Version 5 acceleration uses m/s^2, including the recorded X gravity offset. Reject
unverified older feature versions rather than silently mixing units.

Baseline: 767 Hz, uniform grid `start + n/767`, n = 0..7669; at least
ceil(0.99 * 10 * 767) = 7594 in-window observations; span >= 9.99 seconds;
maximum internal/interpolation gap 0.0025 seconds; one continuous segment and
finite XYZ. Adjacent observations outside the window may bracket grid points
only within the same run/segment and gap bound. Reject unsupported endpoints;
never extrapolate, zero-fill, stretch time, or interpolate across segments.
A future online implementation must wait for boundary observations and respect
arrival latency; the offline algorithm is not a zero-look-ahead guarantee.

`data/cnn_cache/<mock|real>/<fingerprint>/run_NNNN/` contains unnormalized
`acceleration.npy`, `auxiliary.npy`, `windows.csv`, `eligibility.csv`, and
`manifest.json`. Each run has its own source/preprocessing fingerprint. Source
hashes cover raw sensor CSV, both feature CSVs, labels and settings. The manifest
also records feature order, configuration, and preprocessing version. Changing
preparation semantics requires incrementing that version. No cache crosses dataset
roots. Preparation streams CSV and retains compact arrays for one run at a time.

The .npy files reserve candidate-window capacity; the first manifest `eligible`
rows are populated and indexed by windows.csv. Unused trailing capacity is never
loaded. Memory maps avoid loading the complete cache. This trades modest disk
space for simpler incremental writing. About 1.59 GB holds all 17,246 candidate
mock acceleration tensors, before metadata. Original recordings remain untouched.

The cache holds input data, not learned CNN features or weights. Every epoch uses
the inputs again but avoids repeated CSV parsing and interpolation. Test caches
are prepared only in explicit final evaluation (or explicit recorded-run prediction).

## Normalization and equal-print training

Fit separate XYZ and 13-feature population Z-score statistics on eligible training
windows only. Compute each run's mean and second moment, then average these moments
equally across training runs. Overlapping windows remain part of this input
population. Compute variance as E[x^2]-E[x]^2 with numerical nonnegativity clipping.
Standard deviation below 1e-8 uses scale 1; affected indices are recorded in the
preprocessing artifact. Do not normalize per window. Validation, test and inference
reuse fitted scales; each CV fold gets independent statistics. Smoke runs fit
statistics only on their selected window subset.

Loss is `BCEWithLogitsLoss` semantics with reduction='none'. For N eligible windows
from R runs, each window from run r receives w=N/(R*n_r). The batch objective is
mean(w * BCE), without division by batch weight sum. This gives equal-run expected
contribution under uniform window shuffling. Epoch reporting sums weighted losses
and divides by N. Validation loss averages per-run mean BCE independently. No
oversampling, undersampling or additional class weighting is used.

## Training configuration

Defaults in models/cnn_1d/config.json:

- AdamW: learning rate 0.001, weight decay 0.0001, betas (0.9,0.999), epsilon 1e-8.
- Batch size 32; maximum 50 epochs; FP32; seed 42; automatic CUDA/CPU selection.
- Four CPU threads by default; DataLoader workers zero for predictable Windows use.
- ReduceLROnPlateau on run-balanced validation loss: factor 0.5, patience 3,
  absolute threshold 0.0001, minimum LR 1e-6. Four consecutive bad epochs trigger
  reduction after a baseline/best observation.
- Early stopping: ten non-improving epochs with absolute minimum improvement
  0.0001. Save every lower-loss checkpoint; restore the best checkpoint afterward.
- Training windows shuffled each epoch; validation/test ordered; final partial
  batches retained. Dropout is disabled for evaluation.
- Output is a logit during training; sigmoid exactly once during inference.
  Score >= 0.5 is failing. Any later threshold tuning must use development data.

Software/hardware can affect exact reproducibility. Save versions, effective
settings, splits, source fingerprints and random states. Current cache loading is
single-process; benchmark before introducing workers or mixed precision.

## Splits, cross-validation and final training

Mock test IDs remain 0003, 0007, 0009, 0012, 0019, 0027 as recorded in
`data/mock_metadata/split.json`. Sorted development run IDs are stratified by label
using four folds, shuffle=True and seed 42. Fold 0 supplies the fixed 18 training /
6 validation split. Four-fold CV rotates the 24 development runs and saves each
fold's exact IDs. No run crosses partitions. Empty eligible runs fail explicitly.
Real datasets require an explicit split JSON with train_validation and test lists;
four-fold operation requires at least four development runs per class.

After non-smoke CV, final_plan.json records the integer median best epoch (minimum
one) and per-epoch median learning rate across folds, carrying a fold's last LR
forward when needed. Final training starts a fresh network on all 24 development
runs, refits normalization, and uses that fixed epoch count/LR schedule without
validation or test-driven stopping. Final testing then evaluates the saved final
model on the reserved six runs. Dataset/config/split mismatches and smoke-derived
final plans are rejected. Do not repeatedly tune against the final test report.

## Evaluation report generation

`evaluate.py` scores the selected checkpoint against recorded labels. It writes
precision, recall, F1 (failing class), balanced accuracy, confusion matrix ordered
[healthy,failing], window counts, evaluated runs and per-run summaries matching
existing RF/XGBoost conventions. Per-run summaries contain true label, fraction
predicted failing and mean score; they are not a separate print classifier.
Undefined metrics are null. Classification metrics pool windows; early stopping
uses run-balanced BCE. Clearly distinguish these weightings.

Fixed training produces validation_predictions.csv and metrics.json. CV saves fold
reports and pooled nozzle_health_oof_predictions.csv plus metrics.json. Final test
reports live in test_evaluation/ to preserve prior experiment records. history.csv
contains epoch training/validation losses and learning rates. No graphical report
is generated in this baseline. Eligibility reports record rejection reasons and
run IDs for retention analysis; auxiliary completeness can reduce acceleration-only
retention. Final numbers must come from preparation rather than assumed audit values.

## Source files, interfaces and artifacts

`cnn_1d.py` is the CLI; model.py defines the network; data.py prepares/loads inputs;
train.py trains and cross-validates; evaluate.py reports metrics; predict.py loads
saved models. requirements-cnn.txt adds PyTorch while preserving recorder requirements.

Run from repository root with the environment containing requirements-cnn.txt:

```powershell
python models/cnn_1d/cnn_1d.py prepare --dataset mock
python models/cnn_1d/cnn_1d.py train --dataset mock --experiment smoke_001 --smoke
python models/cnn_1d/cnn_1d.py train --dataset mock --experiment baseline_001
python models/cnn_1d/cnn_1d.py train --dataset mock --experiment baseline_001 --resume
python models/cnn_1d/cnn_1d.py cross-validate --dataset mock --experiment cv_001
python models/cnn_1d/cnn_1d.py final-train --dataset mock --experiment final_001 --cv-experiment cv_001
python models/cnn_1d/cnn_1d.py evaluate --dataset mock --experiment final_001
python models/cnn_1d/cnn_1d.py predict --dataset mock --experiment final_001 --runs run_0008
```

Use --config for an alternate settings JSON and --split-file for real data.
The prepare command defaults to development runs; --runs can restrict preparation.
Smoke mode uses two training runs and two validation runs (both classes), eight
eligible windows per run, and two epochs. Its metrics are execution checks only.

`experiments/<name>/artifacts/` holds model.pt (state dictionary), config.json,
preprocessing.json, splits.json, provenance.json, selection.json, history.csv,
metrics.json and validation predictions as applicable. Resume checkpoints contain
current weights, optimizer/scheduler state, RNG state and stopping counters. Resume
only trusted local checkpoints; it uses full PyTorch deserialization. Resume
requires the same configuration, split and normalization. CV has fold_0..fold_3
subdirectories plus combined artifacts. New experiments refuse existing names.

Prediction loads saved settings/scales and does not refit them. Outputs preserve
run/window identifiers and UTC boundaries and contain nozzle scores and labels.
Existing run labels are required by the current recorded-run loader. Live unlabeled
inference is a later integration task. Cache and experiment outputs are ignored by
Git. Never delete original data to rebuild a cache.

## Verification and limitations

Run `python -m unittest discover -s tests -v` in the CNN environment. Focused tests
cover shapes, parameter count, gradients, serialization, interpolation, endpoint
support, gaps, segments, invalid XYZ, equal-run normalization/weighting, metric
conventions and scheduler patience. Integration smoke training verifies actual mock
files, artifact writing and checkpoint evaluation. Passing these tests establishes
software behavior, not fault detection accuracy. Full CV and final testing are
separate expensive commands after data correctness is verified.


### Implementation verification record

The initial smoke experiment `smoke_001` prepared development runs 0001, 0002,
0010 and 0005. Exact eligible counts including boundary support were 671/707,
611/647, 614/650 and 400/421 respectively. Training used eight windows per run
from 0001/0002 and validation used eight per run from 0010/0005 for two epochs.
These are smoke execution results, not full development or test performance.
Reserved test measurements were not used. Full experiments were subsequently completed; see the results below.


### Completed full mock experiments

The five-stage workflow completed using the approved baseline configuration and
seed 42. No settings were tuned against reserved test measurements.

- Development preparation: 12,968 / 13,804 windows eligible (6,580 healthy,
  6,388 failing); all 24 runs retained.
- `full_fixed_001`: 18 training / 6 validation runs; 3,233 validation windows.
- `full_cv_001`: four run-stratified folds, pooled OOF predictions for all 12,968
  eligible development windows.
- Fixed training and all CV folds selected epoch 1 as best, and stopped after
  epoch 11 under patience 10. Later training reduced training loss while worsening
  validation loss. This baseline generalizes poorly on these mock partitions.
- `full_final_001`: a fresh model trained on all 24 development runs for one epoch
  at LR 0.001, as selected by the agreed median-best-epoch rule. This was the
  development-selected full final fit, not smoke mode.
- Final test evaluation ran once on all six reserved runs, retaining 3,084 windows.

| Evaluation | Precision | Recall | F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Fixed validation | 0.538462 | 0.340688 | 0.417329 | 0.537365 |
| Four-fold pooled OOF | 0.402907 | 0.290701 | 0.337728 | 0.436232 |
| Final held-out mock test | 0.349912 | 0.437454 | 0.388817 | 0.395198 |

Final test confusion matrix (rows true, columns predicted, healthy then failing):
`[[606, 1111], [769, 598]]`. Threshold remained 0.5. Test predictions, scores,
window identities and metrics were checked for consistency. These results do not
establish real-world nozzle detection and must not be compared directly against
the existing RF/XGBoost results on different data and evaluation partitions.

Reports are under each experiment's artifacts/metrics.json; the final test report
is full_final_001/test_evaluation/metrics.json with test_predictions.csv. Final
weights and fitted preprocessing are in full_final_001/artifacts/. No additional
training or threshold tuning was performed after examining test results.

### XGBoost mock migration verification

The initial migrated baseline completed four CV fits and one final development fit.
OOF evaluation retained 13,804 windows; final test evaluation retained 3,442.
OOF precision/recall/F1/balanced accuracy: 0.496800 / 0.543783 / 0.519231 / 0.500459.
Test precision/recall/F1/balanced accuracy: 0.356145 / 0.607208 / 0.448961 / 0.305820.
Saved-model prediction on run_0008 succeeded; this run belongs to final training.
Checks verified partition isolation, unique window identities, report counts,
all-missing abstention, partial-input scoring, empty evaluation handling and invalid
schema/run-path rejection. No settings were tuned after inspecting test results.
These mock results do not demonstrate real-world fault detection.
