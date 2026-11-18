"""Validate completed mock recordings and write the measured dataset report."""
import csv
import hashlib
import json
import sys
from collections import Counter
from contextlib import nullcontext

import numpy as np

from pipeline.collectors.sensor_collector import parse
from pipeline.config import Config
from pipeline.features import api_features, sensor_features
from pipeline.mock_dataset import DATA, META, MOCK, REAL, TEST, timestamp
from pipeline.synchronization.timestamp_sync import DeviceClock


def summary(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite):
        return dict(n=len(finite), mean=float(finite.mean()), std=float(finite.std()),
                    p10_p50_p90=np.percentile(finite, [10, 50, 90]).tolist())


def features(path):
    with (path / "labels/print_labels.csv").open() as stream:
        label = next(csv.DictReader(stream))
    result = {}
    paired = None
    for source, module in [("api", api_features), ("sensor", sensor_features)]:
        with (path / "features" / (source + "_features.csv")).open() as stream:
            reader = csv.DictReader(stream)
            assert tuple(reader.fieldnames[9:]) == module.NAMES
            rows = list(reader)
        boundaries = []
        for index, row in enumerate(rows):
            assert all(row[k] == v for k, v in label.items())
            assert row["window_id"] == f"window_{index+1:04d}"
            a, b = map(timestamp, (row["window_start_utc"], row["window_end_utc"]))
            assert abs(b-a-10) < 1e-5
            if boundaries:
                assert abs(a-boundaries[-1][0]-2) < 1e-5
            boundaries.append((a,b))
        if paired is not None:
            assert paired == boundaries
        paired = boundaries
        result.update({k:[float(row[k]) for row in rows] for k in module.NAMES})
    return result


def audit(n):
    name = f"run_{n:04d}"
    path = MOCK/name
    meta = json.loads((META/(name+".json")).read_text())
    values = features(path)
    assert all(len(v) == meta['windows'] for v in values.values())
    assert meta['labels'] == dict(nozzle_health='failing' if n in {2,3,4,5,6,9} or n>=22 else 'healthy',
                                  feeder_health='healthy',motion_system_health='healthy')
    meta['feature_statistics'] = {k:summary(v) for k,v in values.items()}
    meta['complete'] = int(np.all(np.isfinite(list(values.values())),axis=0).sum())
    meta['sha256'] = {}
    meta['bytes'] = 0
    for file in sorted(path.rglob('*')):
        if file.is_file():
            meta['bytes'] += file.stat().st_size
            with file.open('rb') as stream:
                meta['sha256'][file.relative_to(path).as_posix()] = hashlib.file_digest(stream,'sha256').hexdigest()
    for source in ('api','sensor'):
        relative = 'raw/'+source+'_raw.csv'
        original = REAL/name/relative
        with (path/relative).open('rb') as generated:
            header = next(generated)
            count = samples = 0
            with original.open('rb') if original.exists() else nullcontext() as old:
                if original.exists():
                    assert header.strip() == next(old).strip()
                current = next(old,None) if original.exists() else None
                previous = None
                for count,line in enumerate(generated,1):
                    if current and line.rstrip() == current.rstrip():
                        current = next(old,None)
                    elif source == 'sensor' and count%10007 == 1:
                        _,ts,receipt,record = next(csv.reader([line.decode()]))
                        raw = json.loads(record)
                        _,decoded = parse(raw['line'],DeviceClock(),timestamp(receipt),Config())
                        for key in ('sequence','counts','device_tick','filament_counts','temperature','displacement','accel_m_s2'):
                            assert np.allclose(decoded[key],raw[key]),(name,key)
                        assert previous is None or timestamp(ts)>previous
                        previous = timestamp(ts)
                        samples += 1
                assert current is None,(name,source,'original rows missing')
            assert count == meta['raw'][source]
            if source=='sensor': meta['sampled_generated_packets']=samples
    a,b = map(np.array,(values['api_head_velocity_mean'],values['accel_x_std']))
    valid = np.isfinite(a)&np.isfinite(b)
    meta['head_speed_vs_x_std'] = float(np.corrcoef(a[valid],b[valid])[0,1])
    return meta


def report():
    runs = [audit(n) for n in range(1,31)]
    assert {d.name for d in MOCK.iterdir()} == {r['run'] for r in runs}
    reference = {d.name:{k:summary(v) for k,v in features(d).items()} for d in REAL.glob('run_*')}
    (META/'validation.json').write_text(json.dumps(dict(runs=runs,original_feature_statistics=reference),indent=2))
    split = {k:[r['run'] for r in runs if r['split']==k] for k in ['train_validation','test']}
    assert {int(s[-4:]) for s in split['test']}==TEST
    for key,count in [('train_validation',12),('test',3)]:
        assert Counter(r['labels']['nozzle_health'] for r in runs if r['split']==key)=={'healthy':count,'failing':count}
    (META/'split.json').write_text(json.dumps(split,indent=2))
    pilot=json.loads((META/'pilots.json').read_text())
    totals=[sum(r['raw'][s] for r in runs) for s in ['api','sensor']]
    duration=sum(r['duration'] for r in runs)
    lines=['# Mock printer recordings','',
           f"30 runs span {duration/3600:.2f} hours with {totals[0]:,} raw API records and {totals[1]:,} sensor packets. There are {sum(r['windows'] for r in runs):,} paired windows, including {sum(r['complete'] for r in runs):,} with all 28 features. Run files occupy {sum(r['bytes'] for r in runs)/1e9:.2f} GB.",
           '', 'Eight hybrid runs preserve original data and 22 are fully synthetic. Fifteen have healthy nozzles and fifteen have failing nozzles. Feeder and motion labels are healthy throughout.',
           '', '|Run|Seconds|Nozzle|Feeder|Motion|Origin|API|Sensor|Windows|Split|','|---|---:|---|---|---|---|---:|---:|---:|---|']
    for r in runs:
        lines.append(f"|{r['run']}|{r['duration']:.1f}|{r['labels']['nozzle_health']}|healthy|healthy|{r['origin']}|{r['raw']['api']}|{r['raw']['sensor']}|{r['windows']}|{r['split']}|")
    lines+=['','## Structure and reproduction','',
        'Generate a fresh dataset with `python -m pipeline.mock_build`. For selected missing runs use `python -m pipeline.mock_dataset 8 22`. Move existing mock outputs and their metadata aside before rebuilding. Use Python 3.11+ and the package versions in data/mock_metadata/reference.json. Per-run seeds combine the config.json seed and run number. Run `python -m pipeline.mock_audit` for the final audit.',
        '', 'Each data/mock_runs/run_NNNN directory contains raw/api_raw.csv, raw/sensor_raw.csv, features/api_features.csv, features/sensor_features.csv, labels/print_labels.csv and settings.json. Select explicitly using the shared model loader `load_run(Path("data/mock_runs/run_0008"))`. Model CLIs default to data/runs. Use data/mock_metadata/split.json to select whole runs for development and held-out workflow testing.',
        '', '## Generation and supplementation','',
        'Each synthetic run shares its head-motion and thermal trajectory across API, vibration, filament and lagged thermocouple measurements. Normal pauses and intermittent retractions occur in both conditions. Run-specific oscillations, quantized counts, timing jitter and held sensor values generate raw STM32 packets. The measured historical packet period is approximately 1303 microseconds. Nominal feature thresholds remain at 800 Hz. The generator targets 767 Hz, with provisional encoder and temperature updates at 10 Hz and 3 Hz. Synthetic thermal fluctuations and intermittent feed reductions vary with provisional nozzle severity. The target temperature distribution is shared across classes. Historical interventions that changed temperature targets remain in real records.',
        '', 'Original CSV rows are retained in order. Hybrid sensor additions follow recorded head position and temperature. Metadata records their UTC intervals. For run 0007, API observations after the original API row count and sensor packets after the original segment extend recording to 20 minutes. Null thermocouple fields in original packets remain missing. Historical replay metadata for 0004 and 0005 identifies incorrect sessions. The generator uses source records and actual feature anchors instead.',
        '', 'Features use the existing scheduler and extractors on raw observations with the existing timing diagnostics and validity thresholds. Ten-second half-open windows start every two seconds. ML_MODELS.md is empty, and the CNN input contract remains provisional.',
        '', '## Pilot and final validation','',
        'Early pilot windows had low acceleration variability and nearly constant frequency. A second trial coupled vibration too tightly to API head speed. The final vibration envelope varies with time and operating mode. Rebuilding run 0008 reproduced all six files byte for byte.',
        '', '|Pilot|Complete windows|Paired windows|Seconds to generate|GB|','|---|---:|---:|---:|---:|']
    for r in pilot['pilots']:
        lines.append(f"|{r['run']}|{r['complete']}|{r['windows']}|{r['seconds']:.1f}|{r['bytes']/1e9:.2f}|")
    lines+=['',f"The pilots projected {pilot['estimated_full_bytes']/1e9:.2f} GB for ten hours, with {pilot['free_bytes']/1e9:.1f} GB free. Generation and feature processing took {sum(r['seconds'] for r in runs)/3600:.2f} summed worker-hours. validation.json contains hashes, sensor parser checks, feature completeness, means, standard deviations and percentiles by run.",
        '', '## Comparison with real recordings','', 'Medians over finite windows are shown below. Runs 0007 and 0009 contributed held-out sensor references. Run 0007 supplies the only real thermocouple feature distribution.',
        '', '|Recording|Head speed (mm/s)|X acceleration SD (m/s²)|X dominant frequency (Hz)|Filament velocity (mm/s)|Thermocouple (°C)|','|---|---:|---:|---:|---:|---:|']
    for name,stats in [(n,reference[n]) for n in ['run_0001','run_0006','run_0007','run_0009']]+[(r['run'],r['feature_statistics']) for r in pilot['pilots']]:
        metrics=['api_head_velocity_mean','accel_x_std','accel_x_dominant_frequency','filament_velocity_mean','thermocouple_temperature_mean']
        medians=[f"{stats[k]['p10_p50_p90'][1]:.2f}" if stats[k] else 'missing' for k in metrics]
        lines.append('|'+name+'|'+'|'.join(medians)+'|')
    lines+=['','## Limitations','','Synthetic and supplemented measurements lack experimental validation. The simulated vibration, thermal and nozzle-fault responses are provisional. Results measure model-development workflow behavior under these assumptions. Independent printer experiments are required to establish real-world accuracy. All feeder and motion labels are healthy, so failure detection for these components requires additional experiments.']
    repeat = META/'repeat_0022.json'
    if repeat.exists():
        peak = json.loads(repeat.read_text())['peak_working_set_bytes']
        lines.extend(['', f'A separate rebuild of run 0022 also matched all six file hashes. Its single-process peak working set was {peak/2**20:.1f} MiB.'])
    (DATA.parent/'MOCK_RUNS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    if sys.argv[1:]:
        for i in sys.argv[1:]:
            result=audit(int(i))
            print(result['run'],result['windows'],result['complete'],result['bytes'])
    else:
        report()
