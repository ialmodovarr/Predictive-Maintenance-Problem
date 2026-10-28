#!/usr/bin/env python3
"""
Print logger for the 3-sensor Nucleo stream (ADXL345 + AS5600 + MCP9600).

Works with both firmware versions (it tells them apart automatically):
  800 Hz firmware:  t_us,seq,x,y,z,fil_counts,temp_c   at 921600 baud (default)
  200 Hz firmware:  t_ms,x,y,z,fil_counts,temp_c       at 115200 baud (run with --baud 115200)
This script records every line during a print and, when you stop it, saves:
  1. a CSV with every sample (written live, so nothing is lost if something crashes)
  2. an Excel file with clearly labeled sheets:
       Info          what the run was, when, how many samples, what each column means
       Summary (1 s) one row per second: vibration, filament, temperature
       Charts        temperature, vibration and filament speed over the print
       Raw data      every sample, converted to g / mm / deg C (split into
                     "Raw data 2", "Raw data 3"... past 1,000,000 rows)

Setup (once):   python3 -m pip install pyserial numpy xlsxwriter
Run:            caffeinate -i python3 print_logger.py --label "benchy, PLA, 210C"
                add --units ms2 for acceleration in m/s^2 instead of g
                (caffeinate keeps the Mac awake; leave it plugged in)
Stop:           Ctrl+C when the print is done. The Excel file is written then.

Close any `screen` session first: only one program can use the serial port.
"""
import argparse, bisect, csv, glob, math, os, sys, threading, time
from array import array
from datetime import datetime, timedelta

import numpy as np

COUNTS_PER_G = 256.0          # ADXL345 full-resolution mode
G_TO_MS2 = 9.80665            # 1 g in m/s^2
# Acceleration units, set from --units in main(): scale multiplies g, SUF goes in column names
ACC = {"scale": 1.0, "suf": "g", "label": "g"}
AS5600_COUNTS_PER_TURN = 4096
EXCEL_MAX_ROWS = 1_000_000    # Excel's limit is 1,048,576 rows per sheet
RAW_EXCEL_LIMIT = 2_000_000   # above this, raw samples stay in the CSV only (unless --raw-excel)
TEMP_NONE = -2**31            # stored when the board sends "nan" (thermocouple off)


def find_port():
    """Find the Nucleo's ST-LINK serial port on Mac, Windows or Linux."""
    from serial.tools import list_ports
    found = []
    for p in list_ports.comports():
        text = f"{p.description} {p.manufacturer or ''} {p.product or ''}".lower()
        if p.vid == 0x0483 or "stlink" in text or "st-link" in text or "stmicro" in text:
            found.append(p.device)
    if not found:
        found = sorted(glob.glob("/dev/tty.usbmodem*")) + sorted(glob.glob("/dev/ttyACM*"))
    if not found:
        listed = ", ".join(p.device for p in list_ports.comports()) or "none"
        sys.exit("No Nucleo found. Check the USB cable is a DATA cable. "
                 f"Serial ports seen: {listed}. You can pass one with --port.")
    if len(found) > 1:
        print("More than one board found:", ", ".join(found), "- using", found[0], "(pick another with --port)")
    return found[0]


class ReplaySource:
    """Reads lines from a saved text file instead of the board (for testing)."""
    def __init__(self, path):
        self.lines = open(path, "rb").read().splitlines(keepends=True)
        self.i = 0
    def readline(self):
        if self.i >= len(self.lines):
            time.sleep(0.05)
            return b""
        self.i += 1
        return self.lines[self.i - 1]
    def close(self):
        pass


class Recording:
    def __init__(self, um_per_turn):
        self.mm_per_count = um_per_turn / AS5600_COUNTS_PER_TURN / 1000.0
        self.t = array("q")          # microseconds, continuous across board restarts
        self.seq = array("q")        # sample number, continuous across board restarts
        self.x = array("h"); self.y = array("h"); self.z = array("h")
        self.c = array("q")          # filament counts since THIS recording started (or the blue button)
        self.craw = array("q")       # filament counts exactly as the board sent them
        self.temp = array("i")       # temp in 1/100 deg C, TEMP_NONE if off
        self.fil_base = None         # board count at the start of the recording / after a restart
        self.read_errors = 0
        self.lost = 0                # samples the board reports as lost (gaps in seq)
        self.fifo_overflows = 0
        self.bus_resets = 0          # times the firmware had to reset a stuck I2C bus
        self.segments = [0]          # index where each board run starts (new segment after a restart)
        self.fmt = None              # "us" (800 Hz firmware) or "ms" (200 Hz firmware)
        self.board_resets = 0
        self.zeroed = 0
        self.notes = []              # board messages worth keeping (startup checks, errors)
        self.first_wall = None
        self._t_offset = 0           # added to board time after a restart
        self._wrap = 0               # microsecond counter wraps every ~71 min
        self._seq_offset = 0
        self._last_raw_t = None
        self._last_raw_seq = None
        self.lock = threading.Lock()

    def _restart(self, t_raw):
        """Board restarted (RESET or power blip): keep time and sample number going forward."""
        self._t_offset = (self.t[-1] + 1250 if len(self.t) else 0) - t_raw
        self._wrap = 0
        self._seq_offset = (self.seq[-1] + 1) if len(self.seq) else 0
        self.board_resets += 1
        self.segments.append(len(self.t))
        self.fil_base = None         # board's filament counter restarted at 0 too
        self.notes.append(f"board restarted at sample {len(self.t)}")
        print("  ! board restarted, time axis continued")

    def add_line(self, line, writer):
        parts = line.split(",")
        if len(parts) in (6, 7) and parts[0].isdigit():
            try:
                if len(parts) == 7:                       # 800 Hz firmware: t_us,seq,...
                    fmt = "us"
                    t_raw = int(parts[0]); seq_raw = int(parts[1]); rest = parts[2:]
                else:                                     # 200 Hz firmware: t_ms,...
                    fmt = "ms"
                    t_raw = int(parts[0]) * 1000; seq_raw = None; rest = parts[1:]
                x, y, z, c = int(rest[0]), int(rest[1]), int(rest[2]), int(rest[3])
                tf = float(rest[4])
                tc100 = TEMP_NONE if math.isnan(tf) else int(round(tf * 100))
            except ValueError:
                return
            if self.fmt is None:
                self.fmt = fmt
            if fmt == "us":
                if self._last_raw_seq is not None:
                    if seq_raw < self._last_raw_seq:
                        self._restart(t_raw)
                    else:
                        if t_raw < self._last_raw_t:      # counter wrapped (every ~71 min)
                            self._wrap += 1
                        gap = seq_raw - self._last_raw_seq - 1
                        if gap > 0:
                            self.lost += gap
                self._last_raw_seq = seq_raw
                seq = seq_raw + self._seq_offset
            else:
                if self._last_raw_t is not None and t_raw < self._last_raw_t - 1_000_000:
                    self._restart(t_raw)
                seq = len(self.t)
            self._last_raw_t = t_raw
            t = t_raw + self._wrap * 2**32 + self._t_offset
            if self.fil_base is None:
                # The board counts filament since IT started, which may be long before this
                # recording. Start the recording's filament at 0 instead.
                self.fil_base = c - (self.c[-1] if len(self.c) else 0)
            craw = c
            c = c - self.fil_base
            with self.lock:
                if self.first_wall is None:
                    self.first_wall = datetime.now()
                    self.t0 = t
                self.t.append(t); self.seq.append(seq)
                self.x.append(x); self.y.append(y); self.z.append(z)
                self.c.append(c); self.craw.append(craw); self.temp.append(tc100)
            k = ACC["scale"] / COUNTS_PER_G
            writer.writerow([f"{(t - self.t0) / 1e6:.6f}",
                             f"{x * k:.4f}", f"{y * k:.4f}", f"{z * k:.4f}",
                             f"{c * self.mm_per_count:.3f}",
                             "" if tc100 == TEMP_NONE else f"{tc100 / 100:.2f}",
                             t, seq, x, y, z, craw])
        elif line.startswith("read error"):
            self.read_errors += 1
            if self.read_errors <= 20:
                self.notes.append(line)
        elif line.startswith("i2c bus reset"):
            self.bus_resets += 1
            if self.bus_resets <= 50:
                self.notes.append(f"{line} at sample {len(self.t)}")
            print("  ! I2C bus was stuck and got reset:", line)
        elif line.startswith("fifo overflow"):
            self.fifo_overflows += 1
            if self.fifo_overflows <= 20:
                self.notes.append(f"{line} at sample {len(self.t)}")
        elif "zeroed" in line:
            self.zeroed += 1
            self.fil_base = 0            # board counter is now 0, same as the recording's
            self.notes.append(f"filament zeroed (blue button) at sample {len(self.t)}")
            print("board:", line)
        else:
            if len(self.notes) < 200:
                self.notes.append(line)
            print("board:", line)          # startup messages: I2C scan, IDs, temp check


def csv_header():
    u = ACC["suf"]
    return ["time_s", f"accel_x_{u}", f"accel_y_{u}", f"accel_z_{u}", "filament_mm", "hotend_temp_C",
            "board_t_us", "sample_index", "accel_x_counts", "accel_y_counts", "accel_z_counts", "filament_counts"]


def reader(src, rec, writer, f, stop):
    n = 0
    while not stop.is_set():
        try:
            raw = src.readline()
        except Exception as e:
            print("Serial error:", e); stop.set(); break
        line = raw.decode("ascii", "ignore").strip()
        if not line:
            continue
        rec.add_line(line, writer)
        n += 1
        if n % 200 == 0:
            f.flush()


def status_line(rec, last_n, dt):
    with rec.lock:
        n = len(rec.t)
        if n == 0:
            return "waiting for data... (press the black RESET button on the Nucleo)", n
        k = n - bisect.bisect_left(rec.t, rec.t[-1] - 1_000_000)   # samples in the last second
        k = max(2, min(k, n))
        a = np.array([rec.x[-k:], rec.y[-k:], rec.z[-k:]], dtype=float).T / COUNTS_PER_G * ACC["scale"]
        vib = float(np.sqrt(((a - a.mean(axis=0)) ** 2).sum(axis=1).mean()))
        mm = rec.c[-1] * rec.mm_per_count
        temp = None if rec.temp[-1] == TEMP_NONE else rec.temp[-1] / 100
        elapsed = (rec.t[-1] - rec.t0) / 1e6
        lost, ovf = rec.lost, rec.fifo_overflows
    rate = (n - last_n) / dt if dt > 0 else 0
    h, rem = divmod(int(elapsed), 3600); m, s = divmod(rem, 60)
    return (f"{h:d}:{m:02d}:{s:02d}  {n:>8d} samples  {rate:5.0f}/s  " +
            ("temp    n/a  " if temp is None else f"temp {temp:6.2f} C  ") +
            f"filament {mm:8.1f} mm  vib {vib:.3f} {ACC['label']}  errors {rec.read_errors}  lost {lost}" +
            (f"  overflows {rec.fifo_overflows}" if rec.fifo_overflows else "") +
            (f"  bus resets {rec.bus_resets}" if rec.bus_resets else "")), n


def per_second_summary(ts, ax, ay, az, mm, temp):
    """One row per second of the print (ts = seconds from start). Returns dict of arrays."""
    u = ACC["suf"]
    sec = np.floor(ts).astype(np.int64)
    starts = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1]])
    ends = np.r_[starts[1:], len(ts)] - 1
    n = np.diff(np.r_[starts, len(ts)])
    out = {"time_s": sec[starts].astype(float), "samples": n}
    means = []
    var_sum = np.zeros(len(starts))
    for a in (ax, ay, az):
        s1 = np.add.reduceat(a, starts); s2 = np.add.reduceat(a * a, starts)
        m = s1 / n
        means.append(m)
        var_sum += np.maximum(s2 / n - m * m, 0)
    out[f"vibration_rms_{u}"] = np.sqrt(var_sum)
    dev = np.sqrt(sum((a - np.repeat(m, n)) ** 2 for a, m in zip((ax, ay, az), means)))
    out[f"vibration_peak_{u}"] = np.maximum.reduceat(dev, starts)
    out[f"accel_x_mean_{u}"], out[f"accel_y_mean_{u}"], out[f"accel_z_mean_{u}"] = means
    out["filament_total_mm"] = mm[ends]
    prev_mm = np.r_[mm[0], mm[ends[:-1]]]
    prev_t = np.r_[ts[0], ts[ends[:-1]]]
    span = np.maximum(ts[ends] - prev_t, 1e-3)
    out["filament_speed_mm_s"] = (mm[ends] - prev_mm) / span
    ok = ~np.isnan(temp)
    tsum = np.add.reduceat(np.where(ok, temp, 0.0), starts)
    tcnt = np.add.reduceat(ok.astype(float), starts)
    with np.errstate(invalid="ignore", divide="ignore"):
        out["hotend_temp_C"] = np.where(tcnt > 0, tsum / np.maximum(tcnt, 1), np.nan)
    return out


def write_excel(path, rec, args, csv_path, end_wall):
    import xlsxwriter
    with rec.lock:
        t = np.frombuffer(rec.t, dtype=np.int64).copy()
        seq = np.frombuffer(rec.seq, dtype=np.int64).copy()
        x = np.frombuffer(rec.x, dtype=np.int16).astype(float)
        y = np.frombuffer(rec.y, dtype=np.int16).astype(float)
        z = np.frombuffer(rec.z, dtype=np.int16).astype(float)
        c = np.frombuffer(rec.c, dtype=np.int64).astype(float)
        craw = np.frombuffer(rec.craw, dtype=np.int64).astype(float)
        ti = np.frombuffer(rec.temp, dtype=np.int32).copy()
        segs = list(rec.segments)
    tc = np.where(ti == TEMP_NONE, np.nan, ti / 100.0)
    k = ACC["scale"] / COUNTS_PER_G
    ax, ay, az = x * k, y * k, z * k
    u, ul = ACC["suf"], ACC["label"]
    mm = c * rec.mm_per_count
    n = len(t)
    ts = (t - t[0]) / 1e6
    if rec.fmt == "us":
        # The board stamps samples in small batches, so raw times wobble by ~0.5 ms.
        # The sensor's own clock is steady, so fit time vs sample number for each board run
        # and use that: evenly spaced, and still locked to the board's clock.
        bounds = [i for i in segs if i < n] + [n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if b - a >= 2:
                p = np.polyfit(seq[a:b] - seq[a], t[a:b] - t[a], 1)
                ts[a:b] = (t[a] - t[0] + np.polyval(p, seq[a:b] - seq[a])) / 1e6
    summ = per_second_summary(ts, ax, ay, az, mm, tc)
    dur = ts[-1]

    wb = xlsxwriter.Workbook(path, {"constant_memory": True})
    bold = wb.add_format({"bold": True})
    head = wb.add_format({"bold": True, "bg_color": "#DDE6F3", "border": 1, "text_wrap": True, "valign": "top"})
    title = wb.add_format({"bold": True, "font_size": 14})
    f3 = wb.add_format({"num_format": "0.000"}); f2 = wb.add_format({"num_format": "0.00"})
    f4 = wb.add_format({"num_format": "0.0000"}); f1 = wb.add_format({"num_format": "0.0"})
    wrap = wb.add_format({"text_wrap": True, "valign": "top"})

    # ---------------- Info ----------------
    ws = wb.add_worksheet("Info")
    ws.set_column(0, 0, 26); ws.set_column(1, 1, 90)
    r = 0
    ws.write(r, 0, "Print sensor log", title); r += 2
    h, rem = divmod(int(dur), 3600); m, s = divmod(rem, 60)
    rows = [
        ("Run label", args.label or "(none given)"),
        ("Started", rec.first_wall.strftime("%Y-%m-%d %H:%M:%S")),
        ("Ended", end_wall.strftime("%Y-%m-%d %H:%M:%S")),
        ("Duration", f"{h}:{m:02d}:{s:02d}  ({dur:.1f} s)"),
        ("Samples", f"{n}  (average {n / max(dur, 1e-3):.1f} per second; " +
                    ("800 Hz firmware)" if rec.fmt == "us" else "200 Hz firmware)")),
        ("Lost samples", f"{rec.lost}  (gaps in the board's sample numbers; FIFO overflows: {rec.fifo_overflows})"),
        ("Read errors", f"{rec.read_errors}  (lines where at least one sensor didn't answer)"),
        ("Board restarts", str(rec.board_resets)),
        ("I2C bus resets", f"{rec.bus_resets}  (the firmware unstuck the sensor bus by itself; a few samples are lost each time)"),
        ("Filament zeroed", f"{rec.zeroed} time(s) with the blue button"),
        ("Acceleration units", "g (1 g = 9.80665 m/s^2)" if u == "g" else "m/s^2 (converted from g with 1 g = 9.80665 m/s^2)"),
        ("Filament calibration", f"{args.um_per_turn} um per wheel turn (AS5600_UM_PER_TURN), {rec.mm_per_count * 1000:.3f} um per count"),
        ("Full-rate CSV", os.path.abspath(csv_path) + "  (every sample; open in Python/pandas for long runs)"),
    ]
    for k, v in rows:
        ws.write(r, 0, k, bold); ws.write(r, 1, v); r += 1
    r += 1
    ws.write(r, 0, "Column meanings", title); r += 1
    glossary = [
        ("time_s", "Seconds since the first sample of this run (from the Nucleo's clock). With the 800 Hz firmware it is evenly spaced, fitted to the sample numbers."),
        ("sample_index", "Sample number from the board (800 Hz firmware). A jump of more than 1 means samples were lost there."),
        ("clock_time", "Wall-clock time on the laptop (Summary sheet only)."),
        (f"accel_x/y/z_{u}", f"ADXL345 acceleration in {ul} on the sensor's own axes (not yet rotated to printer X/Y/Z). Includes gravity, so the axis pointing up reads about " + ("1 g." if u == "g" else "9.81 m/s^2.")),
        (f"vibration_rms_{u}", f"How much the print head shook in that second, in {ul}: gravity and slow tilt removed, then RMS of the 3 axes combined."),
        (f"vibration_peak_{u}", f"Biggest single jolt in that second, in {ul} (same gravity-removed magnitude)."),
        ("filament_mm / filament_total_mm", "AS5600 wheel: total filament that moved through the sensor since this recording started (or since the blue button). Negative steps are retractions. Sign depends on how the sensor is mounted."),
        ("filament_speed_mm_s", "Filament feed rate over that second (change in filament_total_mm divided by time)."),
        ("hotend_temp_C", "MCP9600 thermocouple tip temperature in deg C (read 10 times per second, repeated in between). Blank when the thermocouple is switched off in the firmware."),
        ("*_counts, board_t_us", "Raw values exactly as the board sent them, kept so the data can be recalibrated later. 256 accel counts = 1 g, 4096 filament counts = 1 wheel turn."),
    ]
    for k, v in glossary:
        ws.write(r, 0, k, bold); ws.write(r, 1, v, wrap); r += 1
    if rec.notes:
        r += 1
        ws.write(r, 0, "Board messages", title); r += 1
        for note in rec.notes[:200]:
            ws.write(r, 1, note); r += 1

    # ---------------- Summary (1 s) ----------------
    ss = wb.add_worksheet("Summary (1 s)")
    cols = [("time_s", "time_s", f1, 9), ("clock_time", None, None, 11), ("samples", "samples", None, 9),
            (f"vibration_rms_{u}", f"vibration_rms_{u}", f4, 15), (f"vibration_peak_{u}", f"vibration_peak_{u}", f4, 15),
            (f"accel_x_mean_{u}", f"accel_x_mean_{u}", f4, 14), (f"accel_y_mean_{u}", f"accel_y_mean_{u}", f4, 14),
            (f"accel_z_mean_{u}", f"accel_z_mean_{u}", f4, 14), ("filament_total_mm", "filament_total_mm", f2, 15),
            ("filament_speed_mm_s", "filament_speed_mm_s", f3, 16), ("hotend_temp_C", "hotend_temp_C", f2, 13)]
    for j, (name, _, _, w) in enumerate(cols):
        ss.set_column(j, j, w)
    ss.set_row(0, 30)
    ss.freeze_panes(1, 0)
    for j, (name, _, _, _) in enumerate(cols):
        ss.write(0, j, name, head)
    m_rows = len(summ["time_s"])
    for i in range(m_rows):
        for j, (name, key, fmt, _) in enumerate(cols):
            if name == "clock_time":
                ss.write(i + 1, j, (rec.first_wall + timedelta(seconds=float(summ["time_s"][i]))).strftime("%H:%M:%S"))
            elif fmt is None:
                ss.write_number(i + 1, j, int(summ[key][i]))
            else:
                v = float(summ[key][i])
                if v == v:                      # leave NaN (no temperature) blank
                    ss.write_number(i + 1, j, v, fmt)

    # ---------------- Charts ----------------
    cs = wb.add_worksheet("Charts")
    sheet = "'Summary (1 s)'"
    last = m_rows
    def chart(col, name, ytitle, color):
        ch = wb.add_chart({"type": "scatter", "subtype": "straight"})
        ch.add_series({"name": name, "categories": [ "Summary (1 s)", 1, 0, last, 0],
                       "values": ["Summary (1 s)", 1, col, last, col],
                       "line": {"width": 1.25, "color": color}})
        ch.set_title({"name": name}); ch.set_legend({"none": True})
        ch.set_x_axis({"name": "time (s)", "min": 0})
        ch.set_y_axis({"name": ytitle, "major_gridlines": {"visible": True, "line": {"color": "#E5E5E5"}}})
        ch.set_size({"width": 900, "height": 300})
        return ch
    cs.insert_chart("A1", chart(10, "Hotend temperature", "deg C", "#C0392B"))
    cs.insert_chart("A17", chart(3, "Vibration (RMS per second)", ul, "#2C3E50"))
    cs.insert_chart("A33", chart(9, "Filament speed", "mm/s", "#2E86C1"))
    cs.insert_chart("A49", chart(8, "Filament total", "mm", "#1E8449"))

    # ---------------- Raw data ----------------
    if n > RAW_EXCEL_LIMIT and not args.raw_excel:
        args.summary_only = True
        print(f"{n} samples is too many for a comfortable Excel file: raw samples stay in the CSV "
              "(use --raw-excel to force them in).")
    if args.summary_only:
        print("Skipping raw sheets; every sample is in the CSV.")
    else:
        f5 = wb.add_format({"num_format": "0.00000"})
        raw_cols = [("time_s", ts, f5, 11), (f"accel_x_{u}", ax, f4, 12), (f"accel_y_{u}", ay, f4, 12),
                    (f"accel_z_{u}", az, f4, 12), ("filament_mm", mm, f3, 12), ("hotend_temp_C", tc, f2, 13),
                    ("board_t_us", t, None, 13), ("sample_index", seq, None, 12), ("accel_x_counts", x, None, 13), ("accel_y_counts", y, None, 13),
                    ("accel_z_counts", z, None, 13), ("filament_counts", craw, None, 14)]
        sheets = max(1, -(-n // EXCEL_MAX_ROWS))
        for k in range(sheets):
            a0, a1 = k * EXCEL_MAX_ROWS, min(n, (k + 1) * EXCEL_MAX_ROWS)
            rs = wb.add_worksheet("Raw data" if k == 0 else f"Raw data {k + 1}")
            for j, (_, _, _, w) in enumerate(raw_cols):
                rs.set_column(j, j, w)
            rs.freeze_panes(1, 0)
            for j, (name, _, _, _) in enumerate(raw_cols):
                rs.write(0, j, name, head)
            print(f"  writing raw sheet {k + 1}/{sheets} ({a1 - a0} rows)...")
            for i in range(a0, a1):
                rr = i - a0 + 1
                for j, (_, arr, fmt, _) in enumerate(raw_cols):
                    if fmt is None:
                        rs.write_number(rr, j, int(arr[i]))
                    else:
                        v = float(arr[i])
                        if v == v:
                            rs.write_number(rr, j, v, fmt)
    wb.close()


def main():
    ap = argparse.ArgumentParser(description="Record the 3-sensor stream during a print and save it to Excel.")
    ap.add_argument("--label", default="", help='what this print is, e.g. "benchy, PLA, 210C"')
    ap.add_argument("--port", default=None, help="serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=921600, help="921600 for the 800 Hz firmware (default), 115200 for the old 200 Hz one")
    ap.add_argument("--out", default=None, help="base file name (default: print_<date_time>)")
    ap.add_argument("--duration", type=float, default=None, help="stop by itself after N seconds")
    ap.add_argument("--um-per-turn", type=int, default=50270, help="AS5600 wheel calibration (match the firmware)")
    ap.add_argument("--units", choices=["g", "ms2"], default="g", help="acceleration units: g (default) or ms2 for m/s^2")
    ap.add_argument("--raw-excel", action="store_true", help=f"put raw samples in Excel even above {RAW_EXCEL_LIMIT:,} rows")
    ap.add_argument("--summary-only", action="store_true", help="leave raw samples out of the Excel file (still in the CSV)")
    ap.add_argument("--replay", default=None, help=argparse.SUPPRESS)   # test with a saved text file
    args = ap.parse_args()
    if args.units == "ms2":
        ACC.update(scale=G_TO_MS2, suf="m_s2", label="m/s^2")

    base = args.out or datetime.now().strftime("print_%Y%m%d_%H%M%S")
    base = os.path.splitext(base)[0]
    csv_path, xlsx_path = base + ".csv", base + ".xlsx"

    if args.replay:
        src = ReplaySource(args.replay); where = args.replay
    else:
        import serial
        port = args.port or find_port()
        src = serial.Serial(port, args.baud, timeout=0.5); where = port
    print(f"Reading {where}, saving every sample to {csv_path}")
    print("Press the black RESET button on the Nucleo to see the startup checks. Ctrl+C to stop.\n")

    rec = Recording(args.um_per_turn)
    f = open(csv_path, "w", newline="")
    writer = csv.writer(f)
    writer.writerow(csv_header())
    stop = threading.Event()
    th = threading.Thread(target=reader, args=(src, rec, writer, f, stop), daemon=True)
    th.start()

    t_start = time.time(); last_n = 0; last_t = time.time()
    try:
        while not stop.is_set():
            time.sleep(2 if not args.replay else 0.2)
            now = time.time()
            msg, last_n = status_line(rec, last_n, now - last_t)
            last_t = now
            print(msg)
            if args.duration and now - t_start > args.duration:
                break
            if args.replay and src.i >= len(src.lines):
                break
    except KeyboardInterrupt:
        print("\nStopping...")
    stop.set()
    th.join(timeout=2)
    end_wall = datetime.now()
    f.flush(); f.close(); src.close()

    if len(rec.t) < 2:
        print("No data recorded. Check that the board is running the 800 Hz firmware and that the baud rate matches "
              "(921600 by default; use --baud 115200 for the old 200 Hz firmware). Garbled text also means a baud mismatch.")
        return
    print(f"Saved {len(rec.t)} samples to {os.path.abspath(csv_path)}")
    print("Writing Excel file (a long print can take a minute or two)...")
    write_excel(xlsx_path, rec, args, csv_path, end_wall)
    print(f"Saved {os.path.abspath(xlsx_path)}")


if __name__ == "__main__":
    main()
