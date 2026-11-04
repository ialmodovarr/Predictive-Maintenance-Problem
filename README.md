# Printer Health Dashboard

Streamlit dashboard for our UVA ECE capstone: predictive maintenance for a 3D printer (UltiMaker S3).
It shows live readings from our three sensors and says whether the printer is **HEALTHY** or **FAILING**.

Health is shown per component, one card each:

| Component | Sensor | What the card shows |
|---|---|---|
| Nozzle | MCP9600 + K-type thermocouple | Nozzle temperature (°C) vs target |
| Feeder | AS5600 custom filament motion sensor | Feed speed (mm/s) and total filament |
| Movement system (motor + belt) | ADXL345 accelerometer on the print head | Vibration per axis (RMS, m/s²) |

## Run it

```
python3 -m pip install -r requirements.txt
python3 -m streamlit run dashboard.py
```

Run it from inside this folder so it picks up the colors in `.streamlit/config.toml`.
It opens at http://localhost:8501.

## Data sources (pick in the sidebar)

- **Simulated**: fake data, no printer needed. First 30 s is idle, then printing starts. Use "Inject fault" to test a nozzle, feeder or movement system failure.
- **Replay CSV**: plays back a recording made by `print_logger.py`.
- **Live CSV**: follows the CSV that `print_logger.py` is writing during a print. The logger keeps the serial port and the dashboard just reads its file, so both run at once.

## Plugging in the ML model

Everything goes through one function, `predict()` in `dashboard.py`. Right now it uses simple threshold rules as a placeholder.
To connect the real model, replace the body of `predict()` and return the same thing:

```python
{
    "label": "HEALTHY" or "FAILING",
    "confidence": 0.0 to 1.0, or None,
    "subsystems": {"Nozzle": bool or None, "Feeder": bool, "Movement system": bool},
    "source": "short text shown under the status",
}
```

`True` means that component is failing. Use exactly these three names, since the dashboard cards look them up by name. Use `None` for the nozzle when there's no temperature data.

The input is a DataFrame with one row per second (newest last) and these columns:
`time_s, rms_x, rms_y, rms_z, vib_rms, filament_total_mm, filament_speed_mm_s, temp_c`.
