# Three-Sensor Test Rig: Team Guide

*ADXL345 \+ AS5600 \+ MCP9600 on one STM32 · how to record all sensors at once (Mac \+ Windows)*

**Last updated:** Oct 21, 2026 (Sofia) · **Status:** ADXL345 and AS5600 share one I2C bus on the breadboard, streaming at about 765 samples/s (800 Hz setting). MCP9600 is switched off in the firmware until the replacement arrives.

## What this is

Three sensors watch the UltiMaker S3 while it prints. All three talk to the same Nucleo board (STM32F446RE) over the same two I2C wires. The Nucleo sends one line of numbers 800 times a second over USB to a laptop. A Python script on the laptop records every line during a print and, when you stop it, saves a labeled Excel file plus a CSV with every sample. Those files are the training data for the ML model.

| Sensor | Measures | Where it is | I2C address |
| :---- | :---- | :---- | :---- |
| ADXL345 accelerometer | How the print head shakes (vibration) | On the print head | 0x53 |
| AS5600 magnetic angle sensor | Filament moving into the printer (our flow sensor wheel) | On the filament path at the back of the S3 | 0x36 |
| MCP9600 thermocouple amplifier | Hotend temperature (K-type thermocouple) | Thermocouple at the hotend | 0x67 |

&nbsp;

```
3 sensors --(shared I2C bus on a breadboard)--> Nucleo STM32 --(USB, 921600 baud)--> laptop: print_logger.py --> print_<date>_<time>.xlsx + .csv
```

**Most of you only need Part 2** (record a print). The board keeps its program when unplugged. Part 1 is for checking or redoing the wiring, Part 4 is for reprogramming the board.

This guide builds on the **ADXL345 Team Guide**. The laptop setup and CubeIDE basics are the same; this one covers what changes when all three sensors run together.

### What's in this folder

| File | What it is |
| :---- | :---- |
| `README.md` / `Three_Sensor_Team_Guide.docx` | This guide (same content, two formats) |
| `code/print_logger.py` | Recorder. Runs on your laptop (Mac or Windows), shows a status line, saves Excel \+ CSV |
| `code/requirements.txt` | Python packages the recorder needs |
| `code/firmware/main.c` | The full `main.c` for the board (800 Hz, all three sensors, automatic I2C bus recovery) |
| `code/firmware/adxl345.h`, `adxl345.c` | Accelerometer driver |
| `code/firmware/as5600.h`, `as5600.c` | Filament sensor driver |
| `images/` | Pictures used in this guide |

## Part 1: Wire the sensors

### How three sensors share two wires

I2C is a **bus**: every sensor connects to the same two signal wires, **SDA** (data) and **SCL** (clock). The Nucleo tells them apart by address (0x53, 0x36, 0x67), so nothing conflicts. Power (3.3V) and ground are shared too. That means the Nucleo only uses **4 pins** no matter how many sensors are on the bus.

The breadboard is the "splitter": one row for each of the 4 signals, the Nucleo plugs into each row once, and every sensor plugs into the same rows.

### What you need

- Full-size **solderless** breadboard (the plastic kind you push wires into). A perfboard with separate holes does not work.  
- **Male-to-male jumper wires.** The ADXL345 and AS5600 use **STEMMA QT cables** whose ends are female sockets, and the Nucleo's headers are female too, so every connection needs a male-to-male jumper as an adapter.  
- No powered breadboard or extra power supply. The Nucleo powers everything from its 3V3 pin (the three sensors draw about 10 mA total).

STEMMA QT cable colors are always the same: **red \= 3.3V, black \= GND, blue \= SDA, yellow \= SCL**. Either STEMMA port on a sensor board works (they are identical).

### The Nucleo pins

Same 4 pins as the ADXL345 guide. Use the **inner black headers**, never the double row of pins along the outer edge.

| Wire | Nucleo pin | Where |
| :---- | :---- | :---- |
| Red | **3V3** | Left inner header (POWER), the hole labeled 3V3, just above 5V. **Never 5V.** |
| Black | **GND** | Left inner header, the GND hole just below 5V |
| Yellow | **SCL / D15** (PB8) | Right inner header, top hole |
| Blue | **SDA / D14** (PB9) | Right inner header, 2nd hole |

![Where the 4 wires go on the Nucleo (photo from the ADXL345 guide).](images/nucleo_pins.jpg){width="4.4in"}

### The breadboard map

Hold the breadboard with the numbers running **60 on the left to 1 on the right**, letters **a to e** on the top half. Every hole along a long red **\+** or blue **−** rail is connected. In the middle, each numbered column is two separate groups of 5 holes (a to e, and f to j), split by the center gap.

![Our breadboard: full-size, 60 columns, power rails top and bottom.](images/breadboard.jpg){width="5.2in"}

![Exactly where every wire goes. Dashed wires: the MCP9600, plug in when the replacement arrives.](images/breadboard_wiring.png){width="6.2in"}

| From | To (breadboard) |
| :---- | :---- |
| Nucleo 3V3 (red jumper) | Top **\+** rail, column 2 |
| Nucleo GND (black jumper) | Top **−** rail, column 1 |
| Nucleo D14 / SDA (blue jumper) | **Column 10, hole e** |
| Nucleo D15 / SCL (yellow jumper) | **Column 14, hole e** |
| ADXL345: red / black / yellow / blue | \+ rail / − rail / 14a / 10a |
| AS5600: red / black / yellow / blue | \+ rail / − rail / 14b / 10b |
| MCP9600 (when replaced): VIN / GND / SCL / SDA | \+ rail / − rail / 14c / 10c |

The exact letter does not matter: all 5 holes in column 10 (a to e) are connected, same for column 14\. What matters is that **every SDA wire is in column 10 and every SCL wire is in column 14**, all on the a to e side. Leave columns 11 to 13 empty so SDA and SCL can never touch.

On the MCP9600, read the printed pin labels (VIN, GND, SCL, SDA) and ignore the other pins (3Vo, ADDR, ALERT). If it has STEMMA ports, use a STEMMA cable the same way as the other two.

### Where the sensors sit on the printer

- **ADXL345:** on the top block of the print head, vertical, connector up, cable running up a clear filament tube to the back (details and photos in the ADXL345 guide).  
- **AS5600 flow sensor:** on the filament path at the back of the S3. Give it **its own cable** to the breadboard rather than chaining it through the ADXL345 on the moving head.  
- **MCP9600:** the board stays near the breadboard; only the thermocouple wire goes to the hotend.

![The ADXL345 on the print head.](images/mount_closeup.jpg){width="3.6in"}

### Before you power on

1. **Unplug the Nucleo's USB** before moving any wire. Never unplug a sensor while the board is powered.  
2. **Continuity check** (multimeter, beep mode): \+ rail to − rail must **not** beep. Column 10 to column 14 must **not** beep.  
3. Check every red wire is on **\+** and every black wire is on **−** (top rails).

## Part 2: Record a print

### Step 1\. One-time laptop setup

You need Python 3.9 or newer (see the ADXL345 guide if Python is not installed yet).

|  | Mac (Terminal) | Windows (PowerShell) |
| :---- | :---- | :---- |
| Go to the code folder | `cd ~/Downloads/three-sensor-guide/code` | `cd $HOME\Downloads\three-sensor-guide\code` |
| Install the packages | `python3 -m pip install -r requirements.txt` | `py -m pip install -r requirements.txt` |

The packages are `pyserial` (talks to the board), `numpy` (math) and `xlsxwriter` (writes the Excel file).

### Step 2\. Start the recorder

Close anything else using the board's port first (`screen`, PuTTY, CubeIDE terminal, another copy of the script). Then, from the `code` folder:

|  | Mac (Terminal) | Windows (PowerShell) |
| :---- | :---- | :---- |
| Start recording | `caffeinate -i python3 print_logger.py --label "..."` | `py print_logger.py --label "..."` |
| Same, acceleration in m/s² | add `--units ms2` | add `--units ms2` |
| Pick the port yourself (only if needed) | `... --port /dev/cu.usbmodem1203` | `... --port COM5` |

`caffeinate -i` keeps the Mac awake for as long as the recorder runs. On Windows, set **Sleep \= Never** while plugged in (Settings \> System \> Power). If the laptop sleeps, the recording stops.

Other options:

| Option | What it does |
| :---- | :---- |
| `--label "..."` | Note saved in the Excel Info sheet. Always use it (rules in Part 3\) |
| `--units ms2` | Acceleration in m/s² instead of g (1 g \= 9.80665 m/s²) |
| `--summary-only` | Leave the raw samples out of the Excel file (they are still in the CSV) |
| `--raw-excel` | Force raw samples into Excel even for long prints (see Part 3\) |
| `--duration 600` | Stop by itself after 600 seconds |
| `--baud 115200` | Only for the old 200 Hz firmware. The default (921600) matches the current firmware |

### Step 3\. What you should see

**Press the black RESET button** on the Nucleo once. The startup checks print with `board:` in front:

![Example of a good start (printer idle). Your numbers will differ slightly.](images/mac_run_example.png){width="6.2in"}

| Startup line | Good | If not |
| :---- | :---- | :---- |
| `--- 800 Hz firmware ---` | Present | Missing: the board runs old code. Reflash (Part 4\) |
| `I2C1 scan:` | `0x36 0x53` (plus `0x67` when the MCP9600 is in) | A sensor is missing: check its wires (Part 1\) |
| `ADXL345 DEVID` | `0xE5` | Accelerometer wiring |
| `ADXL345 800 Hz + FIFO setup` | `OK` | Accelerometer wiring |
| `AS5600 STATUS` | `magnet=YES` | Magnet missing or too far from the chip |
| `MCP9600 ...` | `off` for now; later a room temperature | See Part 4.4 |

Then a status line every 2 seconds:

| Field | Meaning | Healthy |
| :---- | :---- | :---- |
| `765/s` | Samples received per second | About 765 (the ADXL345's own clock, see Part 5\) |
| `temp` | Hotend temperature | `n/a` while the MCP9600 is off |
| `filament` | Filament fed since the recording started (or since the blue button) | Starts at 0.0, climbs while the printer extrudes |
| `vib` | Print-head vibration over the last second (gravity removed) | Idle about 0.007 g (0.07 m/s²); printing several times higher |
| `errors` | Lines where a sensor did not answer | 0, or a few |
| `lost` | Samples the board says it could not send | 0, or a few dozen per print |
| `overflows` | Times the accelerometer's buffer filled up (only shown if \> 0). Each one loses a few samples | 0, or a few |
| `bus resets` | Times the board unstuck the I2C bus by itself (only shown if \> 0\) | 0\. A few per print is OK |

### Step 4\. During the print

1. **Let it sit idle at least 30 seconds** before starting the print, so every file starts with a quiet baseline.  
2. **Press the blue USER button** on the Nucleo right when the print starts. Filament already starts at 0 when the recorder starts; the button restarts it at 0 so it counts only the print.  
3. Start the print and **write down the clock time** (and when it finished).  
4. Watch the status line now and then: `filament` should climb once the printer starts extruding, and `errors` / `lost` should stay flat.

### Step 5\. Stop and save

When the print is done and the head is still again, press **Ctrl+C** in the terminal. The recorder then writes the Excel file (about a minute per hour of printing). Both files land in the folder you ran the command from:

- `print_<date>_<time>.csv`: every sample, written live (safe even if something crashes)  
- `print_<date>_<time>.xlsx`: the labeled Excel file (Part 3\)

Upload both to the team Drive folder **Recordings/** and add one line per run to the team log: file name, date, start/end clock time, what was printed, printer state (healthy / which fault), mount.

## Part 3: The data

### Rules for good data

1. **Label every run** with `--label`: what was printed, healthy or which fault, the mount. Example: `--label "cube 20mm, healthy, kapton mount"`.  
2. **Start every file idle** (30 s quiet before the print).  
3. **Press the blue button at print start** so filament starts at 0\.  
4. **Compare runs at the same sample rate only.** At 800 samples/s the sensor sees faster shaking than at 200, so vibration numbers come out higher. Don't mix 200 and 800 samples/s files in one training set.  
5. **Mount changes \= new baseline.** Re-taped or moved a sensor: record a new healthy run and say so in the label.  
6. **One program per port.** Close `screen` / PuTTY / CubeIDE terminal before recording.

### The Excel file

| Sheet | What's in it |
| :---- | :---- |
| **Info** | Label, start/end time, duration, sample count, read errors, lost samples, bus resets, units, and a plain explanation of every column |
| **Summary (1 s)** | One row per second: clock time, vibration RMS and peak, average acceleration per axis, total filament (mm), filament speed (mm/s), hotend temperature. Start here |
| **Charts** | Temperature, vibration, filament speed and total filament over the whole print |
| **Raw data** | Every sample, converted to g (or m/s²), mm and °C, plus the raw counts. Continues on "Raw data 2", "Raw data 3"... past 1,000,000 rows (about 21 minutes at 800/s) |

For prints longer than about 40 minutes (over 2 million samples) the raw samples stay in the CSV only, to keep the Excel file a reasonable size. Info, Summary and Charts are always there.

### The columns

| Column | Meaning |
| :---- | :---- |
| `time_s` | Seconds since the first sample, from the board's clock. Evenly spaced at the sensor's real rate (about 1.31 ms apart) |
| `accel_x_g`, `_y_g`, `_z_g` | Acceleration on the **sensor's** axes (not yet the printer's X/Y/Z). Includes gravity: the axis pointing up reads about 1 g. With `--units ms2` the names end in `_m_s2` |
| `filament_mm` | Filament that moved through the flow sensor since the recording started (or the blue button). Negative steps are retractions |
| `hotend_temp_C` | Thermocouple temperature. Blank while the MCP9600 is off |
| `board_t_us` | Board time in microseconds (raw) |
| `sample_index` | Sample number from the board. **A jump of more than 1 means samples were lost there** |
| `accel_*_counts`, `filament_counts` | Raw values as the board sent them, kept for recalibration. 256 counts \= 1 g; 4096 counts \= 1 wheel turn |

In the Summary sheet, `vibration_rms` is how much the head shook in that second (gravity and slow tilt removed, three axes combined) and `vibration_peak` is the biggest single jolt.

Loading the CSV in Python:

```
import pandas as pd

d = pd.read_csv("print_20261021_160200.csv")
acc = d[["accel_x_g", "accel_y_g", "accel_z_g"]]
shake = acc - acc.rolling(800).mean()   # remove gravity
d["vib_g"] = (shake ** 2).sum(axis=1) ** 0.5   # shaking
gaps = (d.sample_index.diff() > 1).sum()   # lost-sample spots
```

## Part 4: Programming the board

Only needed for a new Nucleo or when the firmware changes. Tool install and first-time CubeMX project setup are in the ADXL345 guide (Part 4); this part lists what is different for three sensors.

### 4.1 CubeMX settings

Open the project's `sensors.ioc` and check:

| Setting | Value |
| :---- | :---- |
| I2C1 mode | I2C, pins **PB8 \= SCL**, **PB9 \= SDA** |
| I2C1 speed | **Fast Mode (400 kHz)** |
| USART2 | Asynchronous, Baud Rate **921600**, 8 bits, no parity, 1 stop bit |
| I2C3 | Can stay enabled (left over from an earlier test); nothing uses it |

Click **GENERATE CODE** after any change.

### 4.2 Add the files

1. In CubeIDE's **Project Explorer**, expand `sensors > Core`. Drag `adxl345.h` and `as5600.h` onto **Inc**, and `adxl345.c` and `as5600.c` onto **Src**. Choose **Copy files**.  
2. Open `Core/Src/main.c`, select all (Cmd+A / Ctrl+A), and paste in the whole of `code/firmware/main.c` from this folder. Save.

The provided `main.c` already matches the CubeMX settings above, and all our code sits between the `USER CODE BEGIN` / `END` markers, so Generate Code keeps it.

### 4.3 Build and flash

1. Click the **hammer** (Build). The Console must end with **0 errors**.  
2. Click the green **Run** button (first time: Run \> Run As \> STM32 C/C++ Application). Wait for **Download verified successfully**.  
3. **Run is what puts the code on the board.** RESET only restarts whatever is already there. If the recorder still shows `t_ms,...` or `-999` lines, the old program is still on the board.

### 4.4 Turning the thermocouple back on

When the replacement MCP9600 is wired in (Part 1):

1. Near the top of `main.c`, change `#define USE_MCP9600 0` to `1`. Build and Run.  
2. Startup should show `I2C1 scan: 0x36 0x53 0x67`, `MCP9600 ID = 0x40` and a room temperature. Pinch the thermocouple tip: `temp` should rise within a second or two.  
3. If the scan shows the MCP9600 at another address (0x60 to 0x66), change `MCP9600_ADDR` to match.  
4. The bus now runs at 400 kHz, and some MCP9600 boards are unreliable that fast. If `errors` or `bus resets` climb with it plugged in, tell the hardware team: the fix is moving it to its own I2C bus.

## Part 5: How the firmware works

| What | How often | Why |
| :---- | :---- | :---- |
| Accelerometer (ADXL345) | **800 samples/s** | Sees vibration up to 400 Hz (half the sample rate). Our Oct 14 test showed vibration right up to the old 100 Hz limit |
| Filament wheel (AS5600) | 200 reads/s | Filament speed changes slowly; 200/s is plenty and keeps the bus free |
| Temperature (MCP9600) | 10 reads/s | Temperature changes over seconds |

- **The ADXL345 times itself.** It measures 800 times a second into its own 32-sample buffer (FIFO). Every 2 ms the Nucleo empties the buffer and sends each sample, so none are missed even if the Nucleo is briefly busy.  
- **The real rate is about 765 samples/s, not exactly 800\.** The ADXL345 runs on its own internal clock, and ours is about 4.5% slow (measured Oct 21: 764 to 765/s, no samples missing). The recorder fits `time_s` to the board's accurate clock, so time in the files is correct. For frequency analysis, use `time_s` (or the measured rate in the Info sheet), never assume 800\.  
- **Range is ±16 g** with the same precision as before (256 counts per g), so hard jolts never max out.  
- **Each line** looks like `t_us,seq,x,y,z,fil_counts,temp_c`. `t_us` is microseconds from the CPU's cycle counter (it wraps every 71 minutes; the recorder handles that). `seq` counts samples so gaps are visible.  
- **Why 921600 baud:** each line is about 37 characters. At 800 lines/s that is about 30,000 characters/s; the old 115200 baud carries only about 11,500. 921600 carries about 92,000, so the link is about one third busy.  
- **Why 400 kHz I2C:** reading 800 accelerometer samples a second at 100 kHz would keep the bus busy about 70% of the time, too close to the limit.  
- **Short timeouts:** a read that fails (usually noise on the long head cable while the printer moves fast) gives up after 5 ms. With the old 100 ms timeout, one failed read stalled the loop long enough for the 32-sample buffer (about 42 ms) to overflow, losing 13 to 30 samples each time.  
- **Automatic bus recovery:** a corrupted I2C transfer (noise on the long cable, or a sensor holding the clock) can leave the STM32's I2C peripheral stuck, so every read fails until a reset. After 3 failed reads in a row, the firmware resets the I2C peripheral, clocks the bus free, re-configures the ADXL345 and keeps going. The recorder counts these as `bus resets`, and the samples missed during the reset show up as `lost`.  
- **What g means:** 1 g is the acceleration of gravity (9.81 m/s²). A vibration of 0.2 g means the head's shaking has a typical acceleration about a fifth of gravity. For maintenance the trend matters more than the number: more vibration than the healthy baseline, or new frequencies, is the early warning.

## Part 6: Troubleshooting

| Symptom | Likely cause | Fix |
| :---- | :---- | :---- |
| Lines like `381869,-999.00,-999.00` or a `t_ms,...` header | Old firmware still on the board | Build **and Run** the new `main.c` (Part 4.3) |
| Garbled characters, or "No data recorded" | Baud mismatch | USART2 must be 921600 in CubeMX, and the recorder at its default. If it is and text is still garbled, use 460800 in both (`--baud 460800`) |
| `could not open port` / `Resource busy` / `Access is denied` | Another program has the port | Close `screen`, PuTTY, CubeIDE terminal, other copies of the script |
| A sensor missing from `I2C1 scan` | Loose jumper, or SDA/SCL in the wrong column | SDA \= column 10, SCL \= column 14, a to e side. Unplug the other sensors and test the missing one alone |
| Nothing in the scan | Power rails or Nucleo wires | Red on \+, black on −, Nucleo on 3V3 (not 5V) |
| Samples stop and `errors` climb fast | I2C bus stuck (fixed in current firmware) | Current firmware recovers by itself. On old firmware, press RESET |
| `bus resets` climbing several per minute | Noise on the bus | Move the head cable away from the stepper motor wires; check jumpers are pushed fully in; unplug the MCP9600 if it is the broken one |
| `filament` stays at 0.0 while printing | Wheel not turning | Filament must run between the wheel and the idler bearing; the idler must press on it |
| `AS5600 ... weak=YES`, AGC \= 128 | Magnet field weak | Usually still works. If filament readings are jumpy, move the magnet closer to the chip |
| `temp 0.00` | MCP9600 not converting (our broken unit did this) | Replacement sensor; `USE_MCP9600 1` (Part 4.4) |
| `lost` jumps by 10 to 30 together with `errors` | A failed I2C read (noise while the head moves fast) | A few per print is fine. Use the current firmware (5 ms timeouts). If it happens often, reroute the head cable away from the motor wires |
| `lost` climbing steadily | Board can't keep up | Should not happen. Send the terminal output to the hardware team |
| `filament` doesn't start at 0.0 | Older recorder version (counted from when the board started) | Use the current `print_logger.py` |
| Mac: `find` / `cp` command never finishes | Searching the whole home folder is slow | Not needed: paste `main.c` in CubeIDE instead |
| Browser saved `print_logger (2).py` | Several downloads | Keep the newest (it mentions `bus resets` inside), rename it `print_logger.py` |

The ADXL345 guide's troubleshooting table covers USB cables, drivers, Python install and CubeIDE build errors.

## Part 7: Known limits and next steps

- **MCP9600 is broken** (reads 0.00). Firmware has it off; replacement on order. Test it at 400 kHz when it arrives (Part 4.4).  
- **800 samples/s files are not comparable to the 200 samples/s files** from before Oct 21\. Collect the ML training set at one rate.  
- **Excel row limit:** about 21 minutes of raw data per sheet; prints over about 40 minutes keep raw data in the CSV only.  
- **Breadboard is for the test rig.** Once the setup is final, soldering the same connections onto a perfboard makes it sturdier on the printer.  
- **Flow sensor magnet is on the weak side** (AGC maxed). Worth improving before relying on filament data for the model.  
- **Sensor axes are the ADXL345's own**, not the printer's. Converting them is the "orient" step in `imu-bringup.md` in our project notes; the raw counts are kept so it can be done later.

## Appendix: full code

All code is in the `code/` folder. The Word version of this guide also prints it in full.
