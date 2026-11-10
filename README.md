# Ultimaker S3 predictive maintenance

## Record a print

Use Python 3.10 or later on the computer connected to the printer network and the STM32 USB serial port. In PowerShell, from the repository root:

```powershell
python -m pip install -r requirements.txt
python run_pipeline.py
```

The printer address and Digest credentials are configured in pipeline/config.py. The STM32 port is detected automatically. Set port in Config to select a specific board. The recorder issues GET requests to the print-job and printer endpoints. Close other programs using the serial port.

Select healthy or failing, select affected components by number, and confirm the displayed labels. Recording continues until Ctrl+C. Watch the API and sensor record counts, connection errors, and window delay in the console. A new directory under data/runs is allocated for each confirmed print. Outputs include the two raw CSVs, two feature CSVs, authoritative label CSV and effective settings.json. The raw CSV columns include run ID, mapped UTC, host receipt UTC, and a JSON record. Sensor records preserve the original ASCII data line, native tick, counts and sequence if supplied. API records retain endpoint and complete decoded payload.

## Configuration

Use `python run_pipeline.py --config settings.json` to override fields from [Config](pipeline/config.py). The following is an example for an older six-field board:

```json
{"printer_url":"http://YOUR-PRINTER-ADDRESS","port":"COM3","baud":115200}
```

The seven-field profile defaults to 921600 baud. The archived logger calls these the 200 Hz and 800 Hz firmware profiles. Those names describe expected serial streams. Verify the actual firmware, independent update rates, axis scale, wheel calibration and feed direction on the installed hardware. The default accel_hz is the operator-specified nominal 800 Hz. FFT frequency bins use measured timestamp spacing. The default 95% FFT count threshold accepts at least 7,600 samples per 10-second window, subject to gap and continuity checks. filament_hz and temp_hz configure maximum acceptable intervals. Repeated filament and temperature fields are reported packet values whose freshness is unverified. The proposed 400 Hz, 10–20 Hz and 3 Hz frequencies require independent measurement. The accelerometer-only test sketch does not establish those rates.

## Operation and verification

[System architecture](PROJECT_ARCHITECTURE.MD) describes concurrent recording, clock mapping, windowing and recovery. [Pipeline specification](DATA_PIPELINE.md) and [feature definitions](FEATURE_DEFINITIONS_PROPOSAL.md) describe outputs and units. Run `python -m unittest discover -s tests -v` for synthetic streams and mock hardware tests. See [the original script map](legacy/README.md) for earlier collection tools.

On hardware, check API authentication, serial format, sustained packet throughput and sequence gaps. Measure delays and compare a common physical timing event across sources to validate UTC alignment. Inspect filament direction during advance/retraction and verify acceleration scaling before training. Compare features with recorded machine behavior and measure lag on a full print.
