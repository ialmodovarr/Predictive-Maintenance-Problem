# RF/XGBoost feature definitions proposal

Status: implemented software conventions. Numerical thresholds remain provisional pending hardware validation.
This supplements DATA_PIPELINE.md without changing its 28 feature names, labels,
window schedule, raw-retention requirements, or CNN scope. All thresholds below
are proposed starting settings, not measured hardware specifications.

## Common conventions

- Select observations in the shared UTC interval [window_start, window_end).
  Duration is 10 seconds; stride is 2 seconds.
- Preserve original observations. Filtering, unit conversion, sorting, and any
  interpolation occur only in derived working data and produce quality metadata.
- Use finite, valid measurements; zero is a valid measurement, not missing data.
  An unavailable feature is blank/NaN, never zero-filled.
- Sort working observations by mapped acquisition time. Do not calculate across
  clock-reset segments. Repeated transport records are identified by native
  sequence/record identity, not by identical measurement values.
- Distinct records with conflicting values at the same timestamp are retained
  raw and flagged; exclude that ambiguous timestamp from the affected feature.
  Equal-valued measurements at distinct genuine sample times remain valid.
- Temperature and acceleration summaries use equal weight per genuine sample.
  Derivative summaries use elapsed-time weights, defined below. This distinction
  is deliberate: different mathematical estimands need different weights.
- API temperature units: degrees C; head position: mm; acceleration: m/s^2
  (counts / counts_per_g * 9.81, with 9.81 m/s^2 subtracted from X for gravity;
  raw counts are kept in sensor_raw.csv); filament
  displacement: mm. Conversion calibration must be recorded and hardware-verified.
- Do not treat every repeated temperature/filament field in a serial packet as a
  proven fresh sensor observation. Preserve packets, but flag unknown update cadence.

## Shared formulas

For valid sample values x_i, sample count n, and mean m:

- Mean: sum(x_i)/n.
- Sample standard deviation: sqrt(sum((x_i-m)^2)/(n-1)); requires n >= 2.
- Slope: sum((t_i-mean(t))*(x_i-m))/sum((t_i-mean(t))^2).
  Times are seconds relative to window start to avoid large UTC numeric offsets.
  Requires at least three valid observations at distinct times for live output.
  This is an unweighted least-squares trend, not an endpoint-only difference.

For interval values v_i with valid durations w_i = t_i-t_(i-1):

- Duration-weighted mean: M = sum(w_i*v_i)/sum(w_i).
- Duration-weighted descriptive standard deviation:
  sqrt(sum(w_i*(v_i-M)^2)/sum(w_i)).

The latter describes variation over observed time; it is not the sample-SD
estimator above. Do not apply an n-1 correction to these duration weights.

## API: six features

| Feature | Proposed definition | Units |
| --- | --- | --- |
| api_extruder_temperature_mean | Sample mean of extruder index 0 current temperature | degrees C |
| api_extruder_temperature_std | Sample SD of those temperatures | degrees C |
| api_extruder_temperature_slope | Least-squares slope of those temperatures against actual timestamps | degrees C/s |
| api_extruder_temperature_error | Sample mean of current minus target, paired within each same printer snapshot | degrees C |
| api_head_velocity_mean | Duration-weighted mean of scalar 3D interval speeds | mm/s |
| api_head_velocity_std | Duration-weighted descriptive SD of those speeds | mm/s |

Head speed is sqrt(dx^2+dy^2+dz^2)/dt. Use adjacent complete X/Y/Z observations
in the original time-ordered stream; never bridge an intervening invalid position,
reset, or excessive gap. Both endpoints must be inside the assigned window.
Do not interpolate positions at window edges or import an outside predecessor.
Consequently, velocity coverage excludes unobserved edge intervals.
At approximately 1 Hz, this is speed inferred from net displacement between
snapshots, not the printer's instantaneous path speed.

Temperature error does not interpolate or carry forward targets from other
snapshots. Keep the experimental-intervention leakage warning in DATA_PIPELINE.md.

## Accelerometer: fifteen features

Apply each definition independently to X, Y, and Z, using that axis's valid samples.

| Feature suffix | Proposed definition | Units |
| --- | --- | --- |
| rms | sqrt(sum(x_i^2)/n), without subtracting the window mean | m/s^2 |
| std | Sample SD | m/s^2 |
| peak_to_peak | max(x_i)-min(x_i) | m/s^2 |
| dominant_frequency | Strongest nonzero one-sided FFT power bin after mean removal and Hann taper | Hz |
| kurtosis | Bias-corrected Fisher excess kurtosis | dimensionless |

RMS is not mean-centered; it measures total axis amplitude. A mean-centered RMS
would be nearly redundant with axis SD: centered_rms = sample_sd * sqrt((n-1)/n).
Gravity is removed only by the fixed 9.81 m/s^2 offset on X (config
x_gravity_offset_m_s2), applied at collection time, so accel_x_rms reflects X
dynamics plus any residual static offset (mount tilt, sensor bias). Y and Z keep
their static components. The offset affects only X RMS; SD, peak-to-peak,
dominant frequency and kurtosis are offset-invariant. No high-pass filtering is
assumed. Runs with feature_version < 5 stored and featurized acceleration in g
(raw key "accel"), with no gravity offset.

For kurtosis, let m2 and m4 be central moments with divisor n and
g2 = m4/(m2*m2)-3. Return
(n-1)/((n-2)*(n-3))*((n+1)*g2+6).
Requires n >= 4 and nonzero variance; constant signals give NaN with a reason.
This matches the Fisher, bias-corrected convention, not Pearson kurtosis.

### Frequency processing

1. Use mapped native acquisition times; subtract the first sample time locally.
2. Require a trustworthy acquisition cadence. Split at clock segments, invalid measurements and gaps exceeding fft_gap_s (default 0.00375 seconds). Select the longest continuous segment, breaking equal-duration ties by earliest start. Require at least 5 seconds. Preserve full-window timestamp-span checks for time-domain statistics.
3. Estimate sample interval as the median positive consecutive time difference.
   Use sequence/timing diagnostics to flag packet-batching distortion rather than
   treating receipt timestamps as acquisition times.
4. For acceptable small jitter/gaps, create an evenly spaced derived grid at that
   interval, entirely within the observed span. Linearly interpolate onto that
   grid; never extrapolate or modify stored raw measurements. Record resampling.
5. Remove the interpolated mean, apply the symmetric Hann taper
   h_j = 0.5*(1-cos(2*pi*j/(N-1))), and take the real FFT.
6. Exclude DC; choose the largest squared-magnitude bin. Resolve exact ties using
   the lowest positive frequency. Frequencies are k/(N*sample_interval).
7. If signal variance is zero or timing is unsuitable, output NaN with a reason.
   Do not invent a minimum physical vibration amplitude without noise calibration.

No zero-padding or sub-bin peak fitting is proposed. A full 10-second record gives
approximately 0.1 Hz bin spacing, not a guarantee of 0.1 Hz physical accuracy.

## Filament: four features

Use calibrated signed cumulative displacement, with positive direction verified
to mean forward feed. A retraction is negative movement; do not clamp it to zero.
For periodic count observations, v_i = (d_i-d_(i-1))/dt.
Use adjacent valid records with both endpoints in the window, excluding resets,
unknown discontinuities, and excessive gaps.

| Feature | Proposed definition | Units |
| --- | --- | --- |
| filament_velocity_mean | Duration-weighted mean of signed interval velocities | mm/s |
| filament_movement_rate_std | Duration-weighted descriptive SD of those same velocities | mm/s |
| filament_movement_fraction_pct | 100 times the duration of moving intervals divided by the scheduled window duration | percent (0–100) |
| filament_longest_no_movement_interval_s | Longest sum of consecutive valid interval durations with unchanged position | seconds |

A change of at least one encoder count in either direction marks the interval between its observations as moving. This estimates activity at the resolution of reported encoder updates. Held readings and encoder jitter can affect both measures. Invalid observations, excessive gaps, and segment boundaries break a stationary run. Unobserved window edges and gaps contribute to neither moving time nor stationary runs. The percentage uses the full window denominator, so interpret it alongside filament_observed_s. Use the same minimum interval count and duration as filament velocity. Both values are missing when support is inadequate or clock segments overlap. With adequate observations and no stationary intervals, the longest duration is zero. A stationary interval can include an intentional pause.

Pulse/event-only streams need a separate documented reconstruction of displacement
and observation availability. Do not interpret an event gap as verified no motion
or apply periodic-sample count thresholds to pulse events.

## Thermocouple: three features

| Feature | Proposed definition | Units |
| --- | --- | --- |
| thermocouple_temperature_mean | Sample mean of genuine temperature observations | degrees C |
| thermocouple_temperature_std | Sample SD | degrees C |
| thermocouple_temperature_slope | Least-squares slope against genuine observation timestamps | degrees C/s |

Never estimate sensor update rate from how often the numeric value changes: a
fresh measurement can legitimately equal its predecessor. If firmware repeats a
held temperature without an update marker, report freshness as unknown; summaries
may describe reported values, but metadata must not claim verified fresh coverage.

## Proposed configurable feature-support settings

| Setting | Initial proposal |
| --- | --- |
| Late-arrival allowance | 0.5 seconds after each window end |
| API temperature / paired-error support | At least 5 valid observations spanning at least 7.5 seconds |
| API slope support | Same API support plus at least 3 distinct times |
| API velocity support | At least 4 valid intervals covering at least 7 seconds; no interval longer than 2.5 seconds |
| Acceleration time-domain support, per axis | At least 32 valid samples, span at least 9 seconds |
| Acceleration count coverage | At least 85% of expected samples once acquisition rate is verified/configured |
| FFT support, per axis | Longest continuous segment spanning at least 5 seconds, at least 256 samples, 95% count coverage relative to its own duration; maximum internal gap 0.00375 seconds |
| Periodic filament feature support | At least 4 valid intervals covering at least 8 seconds; maximum gap 3 verified nominal update intervals |
| Thermocouple support | At least 5 valid observations spanning at least 8 seconds. Slope permits packet-loss gaps up to temp_slope_gap_s (default 1 second), provided the clock mapping remains continuous |

Evaluate criteria using observations valid for the particular feature, including
paired current/target support for temperature error. Counts alone do not establish
temporal coverage. Interval coverage is the union of accepted interval durations
divided by the 10-second window duration; sample span is last time minus first time.
Report both rather than calling sample span complete coverage.

These are conservative development defaults to evaluate on recordings and hardware.
The configured nominal acceleration rate defaults to the operator-specified 800 Hz;
actual FFT spacing is estimated from device timestamps. Other sensor rates remain
unverified. Unknown cadence/freshness is explicitly flagged; do not claim sufficient
coverage based on an invented rate. Emit computable features with uncertainty
metadata; leave features requiring unavailable timing/freshness missing.

If a feature fails its support criteria, leave it NaN with a reason. Still emit
both source rows for every window. Do not assign an overall quality classification
or reject whole windows during live processing. Other computable features remain
available when a feature is missing. Later model
preparation decides window selection, predictor inclusion, and missing-value
handling using the diagnostic metadata. Keep diagnostics separate from predictor
columns by default; the model does not replace timestamp or input validation.

Record sample counts, sample spans, accepted interval coverage, largest gaps,
estimated/verified rates, synchronization/freshness flags, per-feature missing
reasons, resampling flags, and processing delay as metadata, not extra model inputs.
Persist the effective settings and feature-definition version with each run.

Finalize a window once its end plus allowance passes. Samples arriving afterward
remain raw, with lateness recorded; do not rewrite finalized feature rows. On stop,
drain collectors and process elapsed complete windows with available data; omit
partial-duration windows and flag any curtailed late-arrival wait.

## Decisions requiring hardware evidence

- Actual firmware, sensor acquisition/update rates, timestamp and sequence semantics.
- Acceleration counts-to-g calibration, the X-axis gravity alignment assumed by
  the 9.81 m/s^2 offset, filament distance calibration and direction.
- Measured serial latency/drift and an achievable synchronization uncertainty bound.

These do not prevent synthetic verification of the formulas and scheduler.
They do prevent claiming verified real-time hardware performance.

## Mathematical references

These references define numerical conventions; they do not validate the proposed
project thresholds or require adding SciPy as a dependency.

- [NumPy standard deviation and ddof](https://numpy.org/doc/stable/reference/generated/numpy.std)
- [SciPy Fisher/Pearson and bias-corrected kurtosis](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosis.html)
- [SciPy periodogram, windowing, and detrending](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.periodogram.html)

## Timing-discontinuity handling

Forward clock discontinuities invalidate the preceding ambiguous segment in pending
feature buffers. Native observations remain raw. Acquisition gaps start new segments,
and filament derivatives never cross those boundaries. When periodic sensor update
rates are unknown, sensor_gap_s (default 0.25 seconds) limits accepted filament
intervals. Temperature slope uses its separate temp_slope_gap_s limit (default 1 second). This is a configurable development
threshold, not a sensor-frequency claim. Temperature means and standard deviations
remain available from reported observations, while slopes across resets, reconnects, ambiguous remapping, or gaps exceeding
temp_slope_gap_s are missing. Acquisition-gap boundaries preserve the clock mapping
and can be crossed by temperature slopes. The five-observation and eight-second
support requirements remain in force. Other supported features remain available.

FFT diagnostics retain full-window support statistics and add fft_segment_duration_s (actual duration used, null when unavailable) and fft_candidate_duration_s. Full-window FFT formulas are retained on complete continuous windows.
