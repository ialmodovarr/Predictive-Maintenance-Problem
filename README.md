# ADXL345 Accelerometer: Team Guide

*3D printer predictive maintenance capstone · how to record vibration data (Mac + Windows)*

**Last updated:** Oct 16, 2026 (Sofia) · **Status:** working on the lab's UltiMaker S3, streaming 200 samples/s

## What this is

An ADXL345 accelerometer is taped to the S3's print head. It measures how the head shakes while it prints. The Nucleo board (STM32F446RE) reads it and sends the numbers over USB to a laptop. A Python script on the laptop shows the readings live and saves them to a CSV file. Those CSVs are the vibration data for the ML model.

```
ADXL345 on print head --(4-wire cable, I2C)--> Nucleo STM32 --(USB)--> laptop: live_accel.py --> accel_<date>_<time>.csv
```

**Most of you only need Part 1** (record data). The board is already programmed, and it keeps its program when unplugged. Part 3 is only for reprogramming the board or setting up a new one.

### What's in this folder

| File | What it is |
|---|---|
| `README.md` / `ADXL345_Team_Guide.docx` | This guide (same content, two formats) |
| `code/live_accel.py` | Live viewer + recorder. Runs on your laptop (Mac or Windows) |
| `code/requirements.txt` | Python packages the viewer needs |
| `code/firmware/adxl345.h`, `adxl345.c` | Accelerometer driver for the STM32 |
| `code/firmware/main_usercode_blocks.c` | The 4 code blocks to paste into a new CubeMX project's `main.c` |
| `code/firmware/main_team_project.c` | The full `main.c` currently on the board (includes a teammate's thermocouple code too) |
| `images/` | Pictures used in this guide |

## Part 1: Record data (board already programmed)

### Step 1. Check the hardware

The sensor is on the **top block of the print head** (the block the two clear filament tubes come out of), standing vertically with its connector pointing up. Its cable runs up along one of the clear tubes to the back of the printer.

![The ADXL345 on the print head (temporary electrical tape mount).](images/mount_closeup.jpg)

![Cable route: up along the clear filament tube to the back of the printer.](images/mount_cable_route.jpg)

Before every session:

1. **Tap the board's corners** with a fingertip. It must not wiggle. If it does, re-tape it. (It is on electrical tape until our high-temp tape arrives.)
2. **Check the gap to the rod** just below the board. The head slides along that rod, so the board and tape must never touch it.
3. **Check the 4 wires at the Nucleo.** Use the jumper ends in the inner black headers:

| Wire color | Nucleo pin | Where |
|---|---|---|
| Red | **3V3** | Left inner header (POWER), 3rd hole from the top |
| Black | **GND** | Left inner header, 5th hole from the top |
| Yellow | **SCL / D15** | Right inner header, top hole |
| Blue | **SDA / D14** | Right inner header, 2nd hole |

![Where the 4 wires go on the Nucleo.](images/01_nucleo_wiring.jpg)

Never use the 5V or VIN holes for the sensor.

4. **Plug the Nucleo into your laptop** with a USB cable that carries **data**. Many cheap cables only charge. The red **COM** LED next to the USB port tells you which:
   - **Solid red (or flashing green):** good, the laptop sees the board.
   - **Blinking red forever:** no data connection. Use a different cable or adapter.

### Step 2. One-time laptop setup

You need Python 3.9 or newer and three packages.

#### Mac

1. Open **Terminal** (Cmd + Space, type `Terminal`, Enter).
2. Check Python: `python3 --version`. If you get "command not found", install Python from python.org (macOS installer).
3. Go to the `code` folder of this guide and install the packages (change the path if you saved the folder somewhere else):

```bash
cd ~/Downloads/accelerometer-guide/code
python3 -m pip install -r requirements.txt
```

4. With the Nucleo plugged in, check that the Mac sees it:

```bash
ls /dev/tty.usbmodem*
```

![Mac example: install, then confirm the board shows up as /dev/tty.usbmodem....](images/mac_setup.png)

No ST-LINK driver is needed on a Mac.

#### Windows

1. **Install Python** from python.org (Windows installer). On the first installer screen, **tick "Add python.exe to PATH"**, then click Install Now.
2. Open **PowerShell** (Start menu, type `PowerShell`, Enter).
3. Go to the `code` folder and install the packages:

```powershell
cd $HOME\Downloads\accelerometer-guide\code
py -m pip install -r requirements.txt
```

4. With the Nucleo plugged in, check that Windows sees it:

```powershell
py -m serial.tools.list_ports -v
```

You should see a port described as **STMicroelectronics STLink Virtual COM Port**. Note its COM number. You can also find it in **Device Manager > Ports (COM & LPT)**.

![Windows example: install, then list ports. The Nucleo is COM5 here; yours may differ.](images/win_setup.png)

![The same board in Device Manager.](images/win_devmgr.png)

If there is **no STLink port** in Device Manager but the board's LEDs are on, the ST-LINK driver is missing. Installing STM32CubeIDE installs it. Without CubeIDE, install ST's driver package **STSW-LINK009** from st.com and replug the board.

### Step 3. Record a run

Run the recorder from the `code` folder. Always give a `--label` that says what the run is (see the label rules in Part 2).

| | Mac (Terminal) | Windows (PowerShell) |
|---|---|---|
| Go to the folder | `cd ~/Downloads/accelerometer-guide/code` | `cd $HOME\Downloads\accelerometer-guide\code` |
| Start recording | `python3 live_accel.py --label "..."` | `py live_accel.py --label "..."` |
| Pick the port yourself (only if needed) | `... --port /dev/tty.usbmodem1103` | `... --port COM5` |

The script finds the board by itself. Then:

1. A **plot window** opens (check behind other windows, or the Dock / taskbar).
2. **Press the black RESET button** on the Nucleo once. The board's startup lines appear in the terminal, prefixed with `board:`. You want to see `I2C scan: 0x53` and `DEVID = 0xE5`. Recording already started even before you press RESET; RESET just shows you the startup check.
3. **Let it sit idle for at least 30 seconds** before starting a print. That gives a quiet baseline in every file.
4. Start the print on the S3, and **write down the clock time** you pressed start (and when it finished).
5. When the print is done and the head is still again, **close the plot window**. That stops the recording and saves the file.

![Mac example of a full run.](images/mac_run.png)

![Windows example of a full run.](images/win_run.png)

The CSV is saved in the folder you ran the command from, named `accel_<date>_<time>.csv`.

Keep the laptop **plugged in, lid open and awake** the whole time. If it sleeps, the recording stops.

### Step 4. What you should see

**Idle (printer still):** flat lines. With the board mounted vertically, **X sits near +1.0 g** (that's gravity), Y and Z near 0. The black vibration line stays around **0.01 g**.

![Idle, from our Oct 14 recording.](images/live_plot_idle.png)

**Printing:** all three lines shake and the vibration line jumps to about **0.1 to 0.5 g**, higher on fast moves and corners. The title should keep saying **200 samples/s**.

![Printing, same recording.](images/live_plot_printing.png)

A whole print looks like this when you plot the saved file afterwards. Idle, short bursts while the printer heats and levels the bed, a long printing section, then idle again:

![The full Oct 14 test print (33 min). Printing is about 30x the idle noise.](images/run_summary.png)

### Step 5. Save and share

1. Upload the CSV to the team Drive folder **Recordings/** (create it if missing).
2. In the team log, add one line per run: file name, date, start/end clock time, what was printed, printer state (healthy / which fault), mount (electrical tape / high-temp tape).

## Part 2: Rules for good data

1. **Only one program can use the board's port at a time.** Close `screen`, PuTTY, Arduino serial monitor, or the CubeIDE terminal before running `live_accel.py`.
2. **Label every run** with `--label`. Include: what was printed, healthy or which fault, and the mount. Example: `--label "cube 20mm, healthy, kapton mount"`.
3. **Start every file idle.** At least 30 s of quiet before the print starts.
4. **Mount changes = new baseline.** Any time the board is re-taped or moved: record a new healthy baseline, and don't mix data from the two mounts without saying so in the labels. (On Oct 14 the board slowly shifted about 2 to 3 degrees on the electrical tape during one print. The high-temp tape fixes that.)
5. **Slide test after touching anything on the head.** Printer idle, push the head slowly to all four corners and make sure the board and cable don't catch.
6. **Don't unplug the sensor cable while the Nucleo is powered.** Unplug USB first.

## Part 3: The data format (for the ML / pipeline side)

Each CSV looks like this:

```
## cube 20mm, healthy, kapton mount
## started 2026-10-16T10:15:00, raw counts, 256 counts = 1 g
t_ms,x,y,z
186679,262,23,-26
186684,262,24,-24
...
```

| Column | Meaning |
|---|---|
| `t_ms` | Board time in milliseconds since the Nucleo was last reset. Steps of 5 ms = 200 samples/s. Not wall-clock time: the 2nd comment line gives the laptop's clock when recording started. |
| `x`, `y`, `z` | Raw acceleration counts. **Divide by 256 to get g.** Range is about ±4 g (±1024 counts). |

Sensor axes as currently mounted: **X is vertical** (reads about +1 g from gravity). **Y and Z are horizontal** and carry most of the head's motion. These are the sensor's axes, not the printer's X/Y; mapping them to printer axes is the "orient" step in `imu-bringup.md` in our project notes.

Loading a file in Python:

```python
import pandas as pd

d = pd.read_csv("accel_20261016_101500.csv", comment="#")
d["t_s"] = (d.t_ms - d.t_ms.iloc[0]) / 1000      # seconds from start of file
for a in "xyz":
    d[a + "_g"] = d[a] / 256                     # counts -> g
## vibration only (gravity removed): subtract each axis' mean, or high-pass filter
```

Check for gaps: consecutive `t_ms` values should differ by exactly 5. A bigger jump means samples were lost (for example the laptop was busy or asleep).

## Part 4: Programming the board from scratch

Only needed for a new Nucleo, or if the firmware changes. Same steps on Mac and Windows except where noted.

### 4.1 Install the tools

1. **STM32CubeIDE** (we use version 2.2.0) and **STM32CubeMX** (we use 6.18.1), both free from st.com (needs a free myST account).
   - Since CubeIDE 2.0, the pin setup tool is a separate app: that's why you need **CubeMX** too. (Alternative: CubeIDE **1.19.0** still has it built in.)
   - Pick the installer for your system. Mac: "Apple Silicon" or "Intel" (Apple menu > About This Mac). Windows: the Windows installer; say **yes** to the ST-LINK driver.
2. If st.com shows **"403 Forbidden"** when downloading: use Chrome, log in to myST first, and turn off any VPN or iCloud Private Relay.
3. **Mac only:** if macOS says an app "cannot be opened because the developer cannot be verified": right-click the app > Open, or System Settings > Privacy & Security > **Open Anyway**.

### 4.2 Create the project in CubeMX

1. **File > New Project > Board Selector.** In **Commercial Part Number** type `NUCLEO-F446RE`, click the board in the list, click **Start Project**.
2. Software components dialog: leave as is (LED + USER BUTTON), click **OK**. Answer **Yes** to "initialize all peripherals with their default mode".
3. **Pinout & Configuration > Connectivity > I2C1:** set mode to **I2C**. In **Parameter Settings**, set **I2C Speed Mode = Fast Mode** (400 kHz).
4. On the chip picture, check **PB8 = I2C1_SCL** and **PB9 = I2C1_SDA** (top edge of the chip, green). If CubeMX put them on PB6/PB7, click PB8 > I2C1_SCL and PB9 > I2C1_SDA.
5. **Connectivity > USART2:** set mode to **Asynchronous**. Parameter Settings: Baud Rate **115200**, 8 bits, no parity, 1 stop bit. **Don't skip this.** The board template only labels the pins; without this step the build fails with `huart2 undeclared`.
6. **Project Manager** tab: Project Name `sensors`, **Toolchain / IDE = STM32CubeIDE**.
7. Click **GENERATE CODE**, then **Open Project**. It opens in CubeIDE.

### 4.3 Add the code

1. In CubeIDE's **Project Explorer**, expand `sensors > Core`. From your file manager, drag `adxl345.h` onto **Inc** and `adxl345.c` onto **Src**. Choose **Copy files**.
2. Open `Core/Src/main.c`. Open `code/firmware/main_usercode_blocks.c` next to it. Paste each of the 4 blocks **between** the matching CubeMX comments:

| Block | Goes between |
|---|---|
| 1. includes | `/* USER CODE BEGIN Includes */` and `/* USER CODE END Includes */` |
| 2. `__io_putchar` | `/* USER CODE BEGIN 0 */` and `/* USER CODE END 0 */` |
| 3. startup check | `/* USER CODE BEGIN 2 */` and `/* USER CODE END 2 */` |
| 4. sampling loop | `/* USER CODE BEGIN 3 */` and `/* USER CODE END 3 */` (inside `while (1)`) |

Keep those comment lines exactly as CubeMX wrote them. Code outside them, or between renamed markers, is deleted the next time someone clicks Generate Code. Use Cmd+F / Ctrl+F to find each marker.

### 4.4 Build and flash

1. Click the **hammer** (Build). The Console must end with **Build Finished. 0 errors**. Warnings are fine.
2. **Click the project name** `sensors` in Project Explorer, then **Run > Run As > STM32 C/C++ Application**, click **OK** on the configuration window. If it offers an ST-LINK firmware update, accept, then Run again. After the first time, the green Run button is enough.
3. Wait for **Download verified successfully**. The board now runs the program, and keeps it after unplugging.

![What a good build and flash look like in the CubeIDE Console (example).](images/cubeide_console.png)

Then go to Part 1, Step 3.

## Part 5: Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No Nucleo found` / COM LED blinks red forever / Mac: `no matches found: /dev/tty.usbmodem*` | Charge-only USB cable, or bad dongle | Use a data cable. LED should go solid. |
| Windows: no "STLink" port in Device Manager, LEDs on | ST-LINK driver missing | Install STM32CubeIDE or ST's STSW-LINK009 driver, replug |
| `could not open port` / `Access is denied` (Windows) / `Resource busy` (Mac) | Another program has the port | Close screen / PuTTY / CubeIDE terminal / another copy of the script |
| Plot says "Waiting for data" | Board not running the program, or port busy | Press RESET. Check above. Reflash if needed (Part 4) |
| `board: I2C scan:` shows nothing, DEVID `0x00` | Sensor not connected | Check the 4 wires (Step 1). Yellow and blue swapped is the usual one |
| `read error` lines | Noise on the long cable to the head | In CubeMX set I2C1 to **Standard Mode** (100 kHz), regenerate, reflash |
| Sample rate in the title well below 200 | Laptop busy, or old firmware (10/s) | Close heavy apps. Reflash with the current firmware |
| Mac: `pip3 install` worked but `import serial` fails | Packages went to a different Python | Always use `python3 -m pip install ...` |
| Windows: `py` is not recognized | Python not on PATH | Reinstall Python with "Add python.exe to PATH", or use `python` instead of `py` |
| Mac: `can't open file ... live_accel.py` | Not in the right folder, or browser renamed it `live_accel (1).py` | `cd` into the `code` folder; check with `ls` (Mac) / `dir` (Windows) |
| CubeIDE: "selection cannot be launched" | Console was selected, not the project | Click `sensors` in Project Explorer, then Run As > STM32 C/C++ Application |
| CubeIDE: "No ST-LINK detected" | Same as the charge-only cable problem | Data cable, direct port |
| Build: `huart2 undeclared` / `implicit declaration of HAL_UART_Transmit` | USART2 not enabled in CubeMX | Part 4.2 step 5, regenerate |
| Build: `invalid storage class for function 'MX_...'` | A missing `}` in `main.c` | The `while (1)` loop and `main()` each need their closing brace, after `USER CODE END 3` |
| Build: `multiple definition of _write` | Two printf redirects | Keep only one (`__io_putchar` or `_write`) |

## Part 6: Known limits and next steps

- **200 samples/s only sees vibration up to 100 Hz.** The Oct 14 data shows energy right up to that limit, so motor/belt signatures likely sit higher. Next firmware step: 800+ samples/s (faster USB serial plus the ADXL345's built-in buffer).
- **Electrical tape is temporary.** Re-mount with high-temp tape, then record a new baseline.
- **`t_ms` resets** whenever the Nucleo resets. Use the start time in the CSV header for wall-clock time.

## Appendix: full code

Everything below is also in the `code/` folder as files. Copy from the files, not from this document (word processors can change quotes and spacing).

### A. code/live_accel.py

```python
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
```

### B. code/firmware/adxl345.h

```c
#ifndef ADXL345_H
#define ADXL345_H

#include "stm32f4xx_hal.h"   /* Nucleo-F446RE. Change if you use a different STM32 family. */

/* SDO/ALT pin low or unconnected -> 0x53. If the I2C scan shows 0x1D, use (0x1D << 1). */
#define ADXL345_ADDR            (0x53 << 1)

#define ADXL345_REG_DEVID       0x00
#define ADXL345_REG_BW_RATE     0x2C
#define ADXL345_REG_POWER_CTL   0x2D
#define ADXL345_REG_DATA_FORMAT 0x31
#define ADXL345_REG_DATAX0      0x32

HAL_StatusTypeDef adxl345_read_devid(I2C_HandleTypeDef *hi2c, uint8_t *id);
HAL_StatusTypeDef adxl345_init(I2C_HandleTypeDef *hi2c);
HAL_StatusTypeDef adxl345_read_xyz(I2C_HandleTypeDef *hi2c, int16_t *x, int16_t *y, int16_t *z);

#endif
```

### C. code/firmware/adxl345.c

```c
#include "adxl345.h"

static HAL_StatusTypeDef write_reg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t val)
{
    return HAL_I2C_Mem_Write(hi2c, ADXL345_ADDR, reg, I2C_MEMADD_SIZE_8BIT, &val, 1, 100);
}

HAL_StatusTypeDef adxl345_read_devid(I2C_HandleTypeDef *hi2c, uint8_t *id)
{
    return HAL_I2C_Mem_Read(hi2c, ADXL345_ADDR, ADXL345_REG_DEVID, I2C_MEMADD_SIZE_8BIT, id, 1, 100);
}

HAL_StatusTypeDef adxl345_init(I2C_HandleTypeDef *hi2c)
{
    HAL_StatusTypeDef s;

    /* DATA_FORMAT: FULL_RES = 1, range +/-4 g -> about 256 counts per g */
    s = write_reg(hi2c, ADXL345_REG_DATA_FORMAT, 0x09);
    if (s != HAL_OK) return s;

    /* BW_RATE: 200 Hz output rate (0x0B), matches the 200 Hz stream in main.c */
    s = write_reg(hi2c, ADXL345_REG_BW_RATE, 0x0B);
    if (s != HAL_OK) return s;

    /* POWER_CTL: Measure bit on. Without this every reading is 0. */
    return write_reg(hi2c, ADXL345_REG_POWER_CTL, 0x08);
}

HAL_StatusTypeDef adxl345_read_xyz(I2C_HandleTypeDef *hi2c, int16_t *x, int16_t *y, int16_t *z)
{
    uint8_t b[6];
    HAL_StatusTypeDef s = HAL_I2C_Mem_Read(hi2c, ADXL345_ADDR, ADXL345_REG_DATAX0,
                                           I2C_MEMADD_SIZE_8BIT, b, 6, 100);
    if (s != HAL_OK) return s;

    /* Little-endian: low byte first */
    *x = (int16_t)((b[1] << 8) | b[0]);
    *y = (int16_t)((b[3] << 8) | b[2]);
    *z = (int16_t)((b[5] << 8) | b[4]);
    return HAL_OK;
}
```

### D. code/firmware/main_usercode_blocks.c

```c
/* =====================================================================
   ADXL345 streaming: the four blocks to paste into a NEW CubeMX project's
   Core/Src/main.c. Each block goes BETWEEN the matching pair of comments
   that CubeMX already wrote, for example between
       / * USER CODE BEGIN Includes * /   and   / * USER CODE END Includes * /
   Do NOT copy the BEGIN/END lines themselves and do NOT rename them:
   CubeMX only keeps code that sits between its exact markers.
   (Spaces added inside the comment markers above so this note stays a comment.)
   ===================================================================== */


/* ---- 1. between USER CODE BEGIN Includes / END Includes ---- */
#include <stdio.h>
#include "adxl345.h"


/* ---- 2. between USER CODE BEGIN 0 / END 0 ---- */
/* Sends printf output to USART2 = the USB serial port on the Nucleo. */
int __io_putchar(int ch)
{
  HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, 10);
  return ch;
}


/* ---- 3. between USER CODE BEGIN 2 / END 2 (runs once at startup) ---- */
  printf("\r\nI2C scan:");
  for (uint8_t a = 1; a < 128; a++) {
    if (HAL_I2C_IsDeviceReady(&hi2c1, a << 1, 2, 10) == HAL_OK) {
      printf(" 0x%02X", a);
    }
  }
  printf("\r\n");

  uint8_t id = 0;
  adxl345_read_devid(&hi2c1, &id);
  printf("ADXL345 DEVID = 0x%02X (expect 0xE5)\r\n", id);

  if (adxl345_init(&hi2c1) != HAL_OK) {
    printf("adxl345_init FAILED\r\n");
  }
  printf("t_ms,x,y,z\r\n");
  uint32_t next_sample = HAL_GetTick();


/* ---- 4. between USER CODE BEGIN 3 / END 3 (inside while (1)) ---- */
    /* One sample every 5 ms = 200 samples per second. */
    if ((int32_t)(HAL_GetTick() - next_sample) >= 0) {
      next_sample += 5;
      int16_t x, y, z;
      if (adxl345_read_xyz(&hi2c1, &x, &y, &z) == HAL_OK) {
        printf("%lu,%d,%d,%d\r\n", (unsigned long)HAL_GetTick(), x, y, z);
      } else {
        printf("read error\r\n");
      }
    }
```
