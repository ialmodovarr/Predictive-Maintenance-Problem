"""Replay pre-pipeline recordings (data/api_data, data/sensors) into data/runs.

Usage (from the project root):
    python -m pipeline.replay.historical [--manifest data/historical_runs.json] [--dry-run]

Each manifest session becomes one run_XXXX folder, written by the same Store,
Scheduler, sensor parser, DeviceClock and feature extractors as a live recording.
Records are fed in time order with receipt time = mapped time, so nothing is
treated as late. Re-running skips sessions already replayed.

Timing reconstruction (best effort, documented in each run's
replay_provenance.json):
- Wall-clock anchors come from the file names (laptop local time, converted to
  UTC with the manifest time zone).
- API logs before the collector fix only stamp HH:MM. Snapshots within each
  minute are spread evenly across that minute (first minute starts at the file
  start time; last minute uses the median per-snapshot period). Error: ~1 s.
  Logs with HH:MM:SS(.fff) stamps are used as-is.
- Sensor CSVs from print_logger.py contain spurious +2^32 us clock jumps and
  occasional corrupted rows (the logger misread garbled serial lines). The
  native board clock is recovered as board_t_us mod 2^32; rows whose clock or
  sample number disagree with their neighbours are kept in raw storage as
  rejected records and excluded from features. The first valid sample is
  anchored at the file-name time (logger start), so sensor-to-API alignment is
  uncertain by roughly +/-2 s.
"""
from __future__ import annotations

import argparse
import heapq
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline.collectors.sensor_collector import parse
from pipeline.config import Config
from pipeline.runtime import write_window
from pipeline.storage.csv_writer import Store
from pipeline.synchronization.timestamp_sync import DeviceClock, utc
from pipeline.windowing.rolling_window import Scheduler

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT/"data"/"api_data"
SENSOR_DIR = ROOT/"data"/"sensors"
RUNS_DIR = ROOT/"data"/"runs"
DEFAULT_MANIFEST = ROOT/"data"/"historical_runs.json"
PROVENANCE = "replay_provenance.json"

WRAP = 2**32
NOMINAL_SAMPLE_US = 1250            # 800 Hz firmware
# Row validation. The board clock (mod 2^32) is the timing reference.
NEIGHBOUR_CLOCK_US = 100_000        # a valid row's successor is at most this far ahead...
NEIGHBOUR_SEQ = 20                  # ...and at most this many sample numbers ahead
MAX_GAP_US = 600_000_000            # forward clock jumps beyond 10 min are corruption
RESYNC_AFTER_ROWS = 200             # follow a self-consistent run after this many rejections
# Clock and sample-number advance disagreeing by more than this is counted as a
# sample-number jump (diagnostic only; the row is kept).
CLOCK_TOLERANCE_US = 500_000
CLOCK_TOLERANCE_FRACTION = 0.10

Record = tuple[float, float, str, dict[str, Any]]   # (mapped t, receipt t, source, record)

SECTION = re.compile(r"^=+\r?\n(/api/v1/\S+)\r?\n=+\r?\n", re.M)
STAMP = re.compile(r"Laptop timestamp:\s*([0-9:.]+)\s*")


# ---------------------------------------------------------------------------
# API logs
# ---------------------------------------------------------------------------

def file_start(name: str, tz: ZoneInfo) -> datetime:
    """Local start time encoded in a collector/logger file name."""
    m = re.search(r"(\d{4}-\d\d-\d\d)_(\d\d-\d\d-\d\d)", name)
    if m:
        local = datetime.strptime(" ".join(m.groups()), "%Y-%m-%d %H-%M-%S")
    else:
        m = re.search(r"(\d{8}_\d{6})", name)
        if not m:
            raise ValueError(f"No start time in file name: {name}")
        local = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")
    return local.replace(tzinfo=tz)


def read_api_sections(path: Path) -> list[tuple[str, str | None, Any]]:
    """(endpoint, laptop stamp or None, payload or None) for every logged request."""
    text = path.read_text(encoding="utf-8", errors="replace")
    heads = list(SECTION.finditer(text))
    sections = []
    for k, head in enumerate(heads):
        body = text[head.end(): heads[k+1].start() if k+1 < len(heads) else len(text)]
        stamp = STAMP.match(body)
        if not stamp:
            sections.append((head.group(1), None, None))       # HTTP/collector error
            continue
        try:
            payload, _ = json.JSONDecoder().raw_decode(body[stamp.end():])
        except json.JSONDecodeError:
            payload = None                                     # truncated final write
        sections.append((head.group(1), stamp.group(1), payload))
    return sections


def estimate_api_times(stamps: list[str], start: datetime) -> list[float]:
    """Epoch seconds for each stamp. HH:MM stamps are spread evenly within their minute."""
    def at(clock: str) -> datetime:
        parts = clock.split(":")
        seconds = float(parts[2]) if len(parts) == 3 else 0.0
        return start.replace(hour=int(parts[0]), minute=int(parts[1]), second=0,
                             microsecond=0) + timedelta(seconds=seconds)

    if all(s.count(":") == 2 for s in stamps):
        out, day = [], timedelta(0)
        for s in stamps:
            t = at(s) + day
            if out and t.timestamp() < out[-1] - 12*3600:
                day += timedelta(days=1)
                t += timedelta(days=1)
            out.append(t.timestamp())
        return out

    groups: list[list[int]] = []
    for i, s in enumerate(stamps):
        if groups and stamps[groups[-1][0]] == s:
            groups[-1].append(i)
        else:
            groups.append([i])
    interior = [len(g) for g in groups[1:-1]]
    period = 60.0/float(np.median(interior)) if interior else 60.0/max(len(g) for g in groups)

    out = [0.0]*len(stamps)
    day = timedelta(0)
    previous_minute = None
    for k, group in enumerate(groups):
        minute = at(stamps[group[0]]) + day
        if previous_minute is not None and minute < previous_minute - timedelta(hours=12):
            day += timedelta(days=1)
            minute += timedelta(days=1)
        previous_minute = minute
        lo, hi = minute.timestamp(), minute.timestamp() + 60.0
        if k == 0:
            lo = max(lo, start.timestamp())
        if k == len(groups) - 1:
            hi = min(hi, lo + len(group)*period)
        step = (hi - lo)/len(group)
        for j, i in enumerate(group):
            out[i] = lo + (j + 0.5)*step
    return out


def api_records(path: Path, tz: ZoneInfo, stats: dict) -> list[Record]:
    sections = read_api_sections(path)
    usable = [(e, s, p) for e, s, p in sections if s is not None and isinstance(p, dict)]
    stats.update(api_sections=len(sections), api_usable=len(usable),
                 api_errors_or_truncated=len(sections) - len(usable),
                 api_stamp_resolution="HH:MM:SS" if usable and all(s.count(":") == 2 for _, s, _ in usable) else "HH:MM")
    times = estimate_api_times([s for _, s, _ in usable], file_start(path.name, tz))
    return [(t, t, "api", {"endpoint": e, "payload": p,
                           "replay_source": f"{path.name}", "replay_laptop_stamp": s,
                           "replay_time_estimated": stats["api_stamp_resolution"] == "HH:MM"})
            for t, (e, s, p) in zip(times, usable)]


# ---------------------------------------------------------------------------
# Sensor logs
# ---------------------------------------------------------------------------

def measured_sample_period_us(tick: np.ndarray, seq: np.ndarray) -> float:
    """Average board-clock time per sample number over clean neighbouring rows."""
    dt = np.diff(tick) % WRAP
    ds = np.diff(seq)
    clean = (ds > 0) & (ds <= 10) & (dt > 0) & (dt < 100_000)
    return float(dt[clean].sum()/ds[clean].sum()) if clean.any() else float(NOMINAL_SAMPLE_US)


def sensor_records(path: Path, tz: ZoneInfo, config: Config, stats: dict) -> Iterator[Record]:
    """Rebuild 7-field firmware lines and parse them exactly as the live collector does."""
    d = pd.read_csv(path)
    stats.setdefault("sensor_files", {})
    file_stats = stats["sensor_files"][path.name] = {"rows": len(d), "rejected_rows": 0,
                                                     "clock_repaired_rows": 0, "parse_errors": 0}
    if d.empty:
        return
    board = d["board_t_us"].to_numpy(np.int64)
    file_stats["clock_repaired_rows"] = int((board >= WRAP).sum())
    # Plain Python values so everything derived from them stays JSON-serializable.
    tick = (board % WRAP).tolist()
    seq = d["sample_index"].to_numpy(np.int64).tolist()
    counts = d[["accel_x_counts", "accel_y_counts", "accel_z_counts", "filament_counts"]].to_numpy(np.int64).tolist()
    temp = pd.to_numeric(d["hotend_temp_C"], errors="coerce").to_numpy(float).tolist()
    start = file_start(path.name, tz).timestamp()
    period_us = measured_sample_period_us(np.asarray(tick, np.int64), np.asarray(seq, np.int64))
    file_stats["measured_sample_period_us"] = period_us

    clock = DeviceClock(config.clock_jump_tolerance_s, config.sensor_gap_s)
    connection = f"replay:{path.name}"
    file_stats.update(resyncs=0, sample_number_jumps=0)
    n = len(d)

    def consistent(a: int, b: int) -> bool:
        clock_step = (tick[b] - tick[a]) % WRAP
        seq_step = seq[b] - seq[a]
        close = 0 < clock_step <= NEIGHBOUR_CLOCK_US and 0 < seq_step <= NEIGHBOUR_SEQ
        # A genuine acquisition gap advances clock and sample number consistently.
        gap = (0 < clock_step <= MAX_GAP_US and seq_step > 0 and abs(clock_step - seq_step*period_us)
               <= CLOCK_TOLERANCE_US + CLOCK_TOLERANCE_FRACTION*seq_step*period_us)
        return close or gap

    def agrees_with_next(row: int) -> bool:
        """Corrupted rows are isolated: a genuine row agrees with one of its next two rows
        (two, so a genuine row followed by a corrupted one is not rejected too)."""
        if row + 1 >= n:
            return True
        return consistent(row, row+1) or (row + 2 < n and consistent(row, row+2))

    last_tick = last_seq = None
    device_us = 0          # unwrapped, restart-continued device time since first valid row
    received = start
    streak = 0             # consecutive rejected rows
    for row in range(n):
        x, y, z, fil = counts[row]
        temperature = "nan" if not np.isfinite(temp[row]) else f"{temp[row]:.2f}"
        line = f"{tick[row]},{seq[row]},{x},{y},{z},{fil},{temperature}"
        reason = None
        if not agrees_with_next(row):
            reason = "disagrees with the following row (isolated corrupted value)"
        elif last_seq is not None:
            clock_step = (tick[row] - last_tick) % WRAP
            seq_step = seq[row] - last_seq
            if seq[row] < last_seq and seq[row] < 1000:
                clock_step = NOMINAL_SAMPLE_US          # board restart: continue the timeline
            elif clock_step > MAX_GAP_US or seq_step <= 0:
                if streak >= RESYNC_AFTER_ROWS:
                    # A long, self-consistent run after a bad anchor: follow it.
                    file_stats["resyncs"] += 1
                    clock_step = clock_step if clock_step <= MAX_GAP_US else NOMINAL_SAMPLE_US
                else:
                    reason = "board clock or sample number does not advance from the previous valid row"
            elif abs(clock_step - seq_step*period_us) > CLOCK_TOLERANCE_US + CLOCK_TOLERANCE_FRACTION*seq_step*period_us:
                # Clock is the timing reference; the sample number is kept as sent
                # and the pipeline's DeviceClock records the jump as an acquisition gap.
                file_stats["sample_number_jumps"] += 1
            if reason is None:
                device_us += clock_step
                received = start + device_us/1e6
        if reason is not None:
            streak += 1
            file_stats["rejected_rows"] += 1
            yield received, received, "sensor", {"line": line, "error": f"historical row rejected: {reason}",
                                                 "replay_source": f"{path.name}:{row}"}
            continue
        streak = 0
        last_tick, last_seq = tick[row], seq[row]
        try:
            t, record = parse(line, clock, received, config)
            record["segment"] = f"{connection}:{record['segment']}"
            if record["invalidates_segment"] is not None:
                record["invalidates_segment"] = f"{connection}:{record['invalidates_segment']}"
            record["receipt_minus_mapped_s"] = received - t
        except ValueError as exc:
            file_stats["parse_errors"] += 1
            t, record = received, {"line": line, "error": str(exc)}
        record["replay_source"] = f"{path.name}:{row}"
        yield t, received, "sensor", record


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

def already_replayed(name: str) -> Path | None:
    for provenance in RUNS_DIR.glob(f"run_*/{PROVENANCE}"):
        try:
            if json.loads(provenance.read_text(encoding="utf8")).get("session") == name:
                return provenance.parent
        except (OSError, json.JSONDecodeError):
            continue
    return None


def replay_session(session: dict, manifest: dict, config: Config) -> Path:
    tz = ZoneInfo(manifest["timezone"])
    stats: dict[str, Any] = {}
    api_path = API_DIR/session["api_file"]
    streams = [iter(api_records(api_path, tz, stats))]
    streams += [sensor_records(SENSOR_DIR/f, tz, config, stats) for f in session["sensor_files"]]
    anchor = min([file_start(api_path.name, tz).timestamp()] +
                 [file_start(f, tz).timestamp() for f in session["sensor_files"]])
    replay_diag = {"replay": "historical", "replay_session": session["name"]}

    with Store(RUNS_DIR, session["labels"], config) as store:
        scheduler = Scheduler(anchor, config)
        counts = {"api": 0, "sensor": 0}
        windows = 0
        last_t = anchor
        no_delay = lambda end: None     # processing delay is meaningless offline
        for t, received, source, record in heapq.merge(*streams, key=lambda r: r[0]):
            while item := scheduler.take(t):
                write_window(store, item, config, no_delay, replay_diag)
                windows += 1
            store.raw(source, t, received, record)
            scheduler.add(source, t, received, record)
            counts[source] += 1
            last_t = max(last_t, t)
        while item := scheduler.take(last_t, True):
            write_window(store, item, config, no_delay, replay_diag)
            windows += 1

        provenance = {
            "session": session["name"],
            "replayed_at_utc": utc(datetime.now(timezone.utc).timestamp()),
            "api_file": str(api_path.relative_to(ROOT)),
            "sensor_files": [str((SENSOR_DIR/f).relative_to(ROOT)) for f in session["sensor_files"]],
            "timezone_assumed_for_file_names": manifest["timezone"],
            "label_source": manifest["label_source"],
            "anchor_utc": utc(anchor),
            "last_record_utc": utc(last_t),
            "windows": windows,
            "raw_records": counts,
            "timing": {
                "api": "HH:MM stamps spread evenly within each minute (~1 s error)"
                       if stats.get("api_stamp_resolution") == "HH:MM" else "laptop HH:MM:SS stamps",
                "sensor": "board clock (mod 2^32) anchored at the logger file-name time; "
                          "sensor-to-API alignment uncertain by about +/-2 s",
            },
            "stats": stats,
        }
        (store.path/PROVENANCE).write_text(json.dumps(provenance, indent=2), encoding="utf8")
        print(f"{store.path.name}: {session['name']} -> {windows} windows, "
              f"{counts['api']} API + {counts['sensor']} sensor raw records", flush=True)
        return store.path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--config", help="JSON file of pipeline.config.Config settings")
    parser.add_argument("--dry-run", action="store_true", help="list what would be replayed")
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    config = Config.load(args.config)
    for session in manifest["sessions"]:
        existing = already_replayed(session["name"])
        if existing:
            print(f"skip {session['name']}: already replayed as {existing.name}")
        elif args.dry_run:
            print(f"would replay {session['name']} ({session['api_file']}, sensors={session['sensor_files']})")
        else:
            replay_session(session, manifest, config)


if __name__ == "__main__":
    main()
