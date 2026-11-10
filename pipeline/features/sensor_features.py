import math
import numpy as np
from .common import series, temperatures, support, intervals, velocity, missing

AXES = ("x","y","z")
METRICS = ("rms","std","peak_to_peak","dominant_frequency","kurtosis")
FEED = ("filament_velocity_mean","filament_movement_rate_std")
MOVEMENT_PERCENT = "filament_movement_fraction_pct"
LONGEST_IDLE = "filament_longest_no_movement_interval_s"
TEMP = "thermocouple_temperature"
NAMES = tuple(f"accel_{a}_{m}" for a in AXES for m in METRICS) + FEED + (MOVEMENT_PERCENT, LONGEST_IDLE) + tuple(TEMP+"_"+s for s in ("mean","std","slope"))


def acceleration(out, prefix, t, x, c, fft_groups=None):
    if fft_groups is None:
        fft_groups = [(t,x)]
    valid = np.isfinite(x)
    x,t = x[valid],t[valid]
    n = len(t)
    stats = support(t)
    expected = c.accel_hz*c.window
    if n>=max(2,c.accel_count,expected*c.accel_coverage) and stats["span_s"]>=c.accel_span:
        out[prefix+"rms"] = float(np.sqrt(np.mean(x*x)))
        out[prefix+"std"] = float(np.std(x,ddof=1))
        out[prefix+"peak_to_peak"] = float(np.ptp(x))
        centered = x-x.mean()
        m2 = np.mean(centered**2)
        if n>=4 and np.ptp(x)>0 and m2>0:
            g2 = np.mean(centered**4)/m2**2-3
            out[prefix+"kurtosis"] = float((n-1)*((n+1)*g2+6)/((n-2)*(n-3)))
    candidates = []
    for times, values in fft_groups:
        valid_indices = np.flatnonzero(np.isfinite(values))
        times, values = times[valid_indices], values[valid_indices]
        cuts = np.flatnonzero((np.diff(times)>c.fft_gap_s) | (np.diff(valid_indices)>1))+1
        for indices in np.split(np.arange(len(times)),cuts):
            if len(indices):
                candidates.append((times[indices],values[indices]))
    t,x = max(candidates,key=lambda pair:(pair[0][-1]-pair[0][0],-pair[0][0]),default=(np.array([]),np.array([])))
    duration = float(np.ptp(t)) if len(t) else 0.
    stats["fft_segment_duration_s"] = None
    stats["fft_candidate_duration_s"] = duration
    if c.accel_hz and duration>=c.fft_min_span and len(t)>=max(c.fft_count,duration*c.accel_hz*c.fft_coverage) and np.ptp(x)>0:
        relative = t-t[0]
        step = np.median(np.diff(relative))
        length = int(relative[-1]/step)+1
        if 3<=length<=c.buffer_limit:
            y = np.interp(np.arange(length)*step,relative,x)
            power = abs(np.fft.rfft((y-y.mean())*np.hanning(length)))**2
            out[prefix+"dominant_frequency"] = float((np.argmax(power[1:])+1)/(step*length))
            stats["fft_resampled"] = True
            stats["fft_segment_duration_s"] = duration
            stats["estimated_hz"] = float(1/step)
    return stats


def extract(records, c):
    out = dict.fromkeys(NAMES, math.nan)
    reasons = {}
    rows = []
    segments = set()
    flags = set()
    seen = {}
    sequence_groups = {}
    residuals = []
    for t,r in records:
        if "error" in r:
            flags.add("serial_message")
            continue
        if r.get("sync") == "out_of_order":
            flags.add("out_of_order_excluded")
            continue
        flags.add(r["sync"])
        segments.add(r["segment"])
        identity = (r["segment"],r["sequence"],r["device_tick"])
        if identity in seen:
            flags.add("duplicate_record")
            # Conflicting retransmissions share one timestamp so series() masks
            # only the conflicting fields. Raw storage retains both packets.
            t = seen[identity]
        else:
            seen[identity] = t
        if r["sequence"] is not None:
            sequence_groups.setdefault(r["segment"],set()).add(r["sequence"])
        if "receipt_minus_mapped_s" in r:
            residuals.append(r["receipt_minus_mapped_s"])
        rows.append((t,{**r, **dict(zip(AXES,r["accel_m_s2"]))}))
    diag = {"sync_flags":sorted(flags),"segments":len(segments),"freshness":"packet_fields_unverified"}
    diag["sequence_gaps"] = sum(max(s)-min(s)+1-len(s) for s in sequence_groups.values())
    diag["missing_packets_before_records"] = sum(r.get("missing_packets_before",0) for _,r in rows)
    diag["max_receipt_residual_s"] = max(residuals,default=None)
    for axis in AXES:
        t,x = series(rows,axis)
        prefix = f"accel_{axis}_"
        fft_groups = [series([(time,row) for time,row in rows if row["segment"]==segment],axis)
                      for segment in sorted(segments,key=str)] if len(segments)>1 else None
        diag[axis] = acceleration(out,prefix,t,x,c,fft_groups)
        if not c.accel_hz:
            reasons[prefix+"dominant_frequency"] = "acquisition_rate_unverified"
        elif len(x) and np.ptp(x)==0:
            reasons[prefix+"dominant_frequency"] = "constant_signal"
        if len(x) and np.ptp(x)==0:
            reasons[prefix+"kurtosis"] = "constant_signal"
    t,x = series(rows,"temperature")
    diag["reported_temperature"] = temperatures(out,TEMP,t,x,c.temp_count,c.temp_span)
    # Only acquisition-gap boundaries retain the existing device-to-UTC mapping.
    # Resets, reconnects and other unclassified boundaries remain ambiguous.
    previous_segment = None
    ambiguous_clock = False
    for _, row in sorted(rows,key=lambda item:item[0]):
        if previous_segment is not None and row["segment"] != previous_segment:
            if row["sync"] != "acquisition_gap":
                ambiguous_clock = True
        previous_segment = row["segment"]
    if ambiguous_clock:
        out[TEMP+"_slope"] = math.nan
        reasons[TEMP+"_slope"] = "ambiguous_clock_boundary"
    elif (diag["reported_temperature"]["max_gap_s"] or 0)>c.temp_slope_gap_s:
        out[TEMP+"_slope"] = math.nan
        reasons[TEMP+"_slope"] = "temperature_gap_exceeds_slope_limit"
    t,x = series(rows,"displacement")
    gap = c.gap_periods/c.filament_hz if c.filament_hz else c.sensor_gap_s
    groups = [(t,x)] if len(segments)<=1 else [
        series([(time,row) for time,row in rows if row["segment"]==segment],"displacement")
        for segment in segments
    ]
    weights, rates = [], []
    longest_idle = 0.
    spans = []
    for times, positions in groups:
        durations, speeds, accepted = intervals(times,positions,gap,signed=True)
        weights.append(durations)
        rates.append(speeds)
        if len(durations):
            idx = np.flatnonzero(accepted)
            spans.append((times[idx[0]], times[idx[-1]+1]))
        # Runs of adjacent, observed intervals with identical encoder positions.
        idle = accepted & (np.diff(positions)==0)
        edges = np.diff(np.r_[False,idle,False].astype(int))
        starts, ends = np.flatnonzero(edges==1), np.flatnonzero(edges==-1)
        if len(starts):
            longest_idle = max(longest_idle,float(max(times[ends]-times[starts])))
    dt,v = np.concatenate(weights),np.concatenate(rates)
    velocity(out,FEED,dt,v,c.filament_intervals,c.filament_span)
    if math.isfinite(out[FEED[0]]):
        spans.sort()
        if any(left[1]>right[0] for left,right in zip(spans,spans[1:])):
            reasons.update(dict.fromkeys((MOVEMENT_PERCENT,LONGEST_IDLE),"overlapping_clock_segments"))
        else:
            out[MOVEMENT_PERCENT] = float(100*dt[v!=0].sum()/c.window)
            out[LONGEST_IDLE] = longest_idle
    diag["filament_observed_s"] = float(dt.sum())
    diag["filament_samples"] = support(t[np.isfinite(x)])
    diag["filament_intervals"] = len(dt)
    diag["missing"] = missing(out,reasons)
    return out,diag
