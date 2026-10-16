#!/usr/bin/env python3
"""
Live ADXL345 viewer + logger for the Nucleo stream (t_ms,x,y,z in raw counts).

Setup (once):   Mac:      python3 -m pip install pyserial matplotlib numpy
                Windows:  py -m pip install pyserial matplotlib numpy
Run:            Mac:      python3 live_accel.py --label "what this run is"
                Windows:  py live_accel.py --label "what this run is"
The Nucleo's port is found automatically (Mac /dev/tty.usbmodem..., Windows COMx).
If more than one board is plugged in, pick one with --port.

Shows the last 10 s of X/Y/Z (in g) and a vibration level line, and saves every
sample to a CSV named after the start time. Close the plot window (or Ctrl+C) to stop.
"""
import argparse, csv, glob, os, sys, threading, time
from collections import deque
from datetime import datetime

import numpy as np
import serial

COUNTS_PER_G = 256.0  # ADXL345 full-resolution mode


def find_port():
    """Find the Nucleo's ST-LINK serial port on Mac, Windows or Linux."""
    from serial.tools import list_ports
    found = []
    for p in list_ports.comports():
        text = f"{p.description} {p.manufacturer or ''} {p.product or ''}".lower()
        if p.vid == 0x0483 or "stlink" in text or "st-link" in text or "stmicro" in text:
            found.append(p.device)
    if not found:  # fall back to name patterns
        found = sorted(glob.glob("/dev/tty.usbmodem*")) + sorted(glob.glob("/dev/ttyACM*"))
    if not found:
        listed = ", ".join(p.device for p in list_ports.comports()) or "none"
        sys.exit("No Nucleo found. Check that the USB cable is a DATA cable and the board's COM LED "
                 f"is solid (not blinking). Serial ports seen: {listed}. "
                 "You can also pass the port yourself, e.g. --port COM5 or --port /dev/tty.usbmodem1103")
    if len(found) > 1:
        print("More than one board found:", ", ".join(found), "- using", found[0], "(pick another with --port)")
    return found[0]


def reader(ser, buf, writer, f, stop, stats, echo):
    while not stop.is_set():
        try:
            line = ser.readline().decode("ascii", "ignore").strip()
        except serial.SerialException as e:
            print("Serial error:", e); stop.set(); break
        if not line:
            continue
        parts = line.split(",")
        if len(parts) == 4 and parts[0].isdigit():
            try:
                t, x, y, z = (int(p) for p in parts)
            except ValueError:
                continue
            buf.append((t, x, y, z))
            writer.writerow([t, x, y, z])
            stats["n"] += 1
            if stats["n"] % 200 == 0:
                f.flush()
        elif echo:
            print("board:", line)  # startup messages (I2C scan, DEVID, errors)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help="serial port (auto-detected if omitted)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--window", type=float, default=10.0, help="seconds shown on screen")
    ap.add_argument("--label", default="", help="note saved at the top of the CSV")
    ap.add_argument("--out", default=None, help="CSV file name (default: accel_<date_time>.csv)")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds (optional)")
    args = ap.parse_args()

    port = args.port or find_port()
    out = args.out or datetime.now().strftime("accel_%Y%m%d_%H%M%S.csv")
    ser = serial.Serial(port, args.baud, timeout=0.5)
    print(f"Reading {port} at {args.baud} baud, saving to {out}")

    f = open(out, "w", newline="")
    if args.label:
        f.write(f"# {args.label}\n")
    f.write(f"# started {datetime.now().isoformat(timespec='seconds')}, raw counts, {COUNTS_PER_G:.0f} counts = 1 g\n")
    writer = csv.writer(f)
    writer.writerow(["t_ms", "x", "y", "z"])

    buf = deque(maxlen=int(args.window * 1000))  # plenty for 200 Hz
    stop = threading.Event()
    stats = {"n": 0}
    th = threading.Thread(target=reader, args=(ser, buf, writer, f, stop, stats, True), daemon=True)
    th.start()

    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    lx, = ax1.plot([], [], label="X", lw=0.8)
    ly, = ax1.plot([], [], label="Y", lw=0.8)
    lz, = ax1.plot([], [], label="Z", lw=0.8)
    ax1.set_ylabel("acceleration (g)")
    ax1.legend(loc="upper left", ncol=3)
    ax1.grid(alpha=0.3)
    lv, = ax2.plot([], [], color="black", lw=1)
    ax2.set_ylabel("vibration (g RMS)")
    ax2.set_xlabel("seconds ago")
    ax2.grid(alpha=0.3)
    t_start = time.time()

    def update(_):
        if args.duration and time.time() - t_start > args.duration:
            plt.close(fig)
            return lx, ly, lz, lv
        if len(buf) < 10:
            fig.suptitle("Waiting for data... (press RESET on the Nucleo if nothing shows up)")
            return lx, ly, lz, lv
        a = np.array(buf, dtype=float)
        t = (a[:, 0] - a[-1, 0]) / 1000.0
        g = a[:, 1:] / COUNTS_PER_G
        lx.set_data(t, g[:, 0]); ly.set_data(t, g[:, 1]); lz.set_data(t, g[:, 2])
        # vibration = RMS of the movement around the average (gravity removed), 0.25 s windows
        dt = np.median(np.diff(a[:, 0])) / 1000.0 if len(a) > 1 else 0.005
        w = max(5, int(0.25 / max(dt, 1e-3)))
        ac = g - g.mean(axis=0)
        mag2 = (ac ** 2).sum(axis=1)
        if len(mag2) >= w:
            rms = np.sqrt(np.convolve(mag2, np.ones(w) / w, mode="valid"))
            lv.set_data(t[w - 1:], rms)
            ax2.set_ylim(0, max(0.05, rms.max() * 1.2))
        ax1.set_xlim(-args.window, 0)
        lo, hi = g.min(), g.max()
        pad = max(0.1, (hi - lo) * 0.1)
        ax1.set_ylim(lo - pad, hi + pad)
        rate = 1.0 / dt if dt > 0 else 0
        fig.suptitle(f"ADXL345 live  |  {rate:.0f} samples/s  |  {stats['n']} saved to {out}")
        return lx, ly, lz, lv

    anim = FuncAnimation(fig, update, interval=100, cache_frame_data=False)
    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    stop.set()
    th.join(timeout=1)
    f.flush(); f.close(); ser.close()
    print(f"Saved {stats['n']} samples to {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
