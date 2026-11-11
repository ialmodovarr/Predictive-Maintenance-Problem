#!/usr/bin/env python3
"""
Printer health dashboard (Streamlit) for the 3D printer predictive maintenance capstone.

Data sources, picked in the sidebar:
  Printer API  polls the UltiMaker S3's own HTTP API (nozzle + bed temperature, head
               position, print job) about once a second. Needs the laptop on the same
               network as the printer and the API credentials in .streamlit/secrets.toml.
  Live API log follows the newest api_data_*.txt that Bella's data/api.py is writing.
               Run her collector as usual during a print; the dashboard only reads its file,
               so the printer is polled once, by her script.
  API log      replays a log written by data/api.py (api_data_*.txt), so the API view
               can be demoed without the printer.
  Simulated    fake sensor data generated in the app, with buttons to inject a fault.
               Works with no printer, no board and no model. Use it to build and demo the UI.
  Replay CSV   plays back a recording made by print_logger.py, sped up if you like.
  Live CSV     follows the CSV print_logger.py is writing during a print right now.
               (Run the logger as usual; the dashboard only reads its file, so both can run
               at once. Only one program can hold the serial port.)

Model output is binary: HEALTHY or FAILING. Until the ML model exists, predict() below
uses simple threshold rules as a stand-in. To plug in the real model, replace the body
of predict() and keep its inputs and outputs the same.

Setup (once):  python3 -m pip install -r requirements.txt
Run:           python3 -m streamlit run dashboard.py
               (run it from the folder that has the .streamlit/config.toml with the colors;
               without it the app still works, just with Streamlit's default sidebar)
"""
import io
import os
import time

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

import notify
import printer_api

G = 9.80665
APPROX_RATE = 800          # samples per second from the board (real rate is about 765)
BYTES_PER_ROW = 100        # rough size of one line of the logger CSV, used to tail the file

st.set_page_config(page_title="Printer Health", page_icon="🖨️", layout="wide")

# ---------------------------------------------------------------------------
# Look and feel (blue palette). Colors for the sidebar, widgets and font also live in
# .streamlit/config.toml next to this file; this CSS styles the cards and custom pieces.
# ---------------------------------------------------------------------------
NAVY = "#0B1F3A"
BLUE_900 = "#1E3A8A"
BLUE_700 = "#1D4ED8"
BLUE_500 = "#3B82F6"
BLUE_300 = "#93C5FD"
BLUE_100 = "#DBEAFE"
INK = "#0F172A"
MUTED = "#64748B"
RED = "#DC2626"
FONT = "Plus Jakarta Sans"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
header[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 2rem; max-width: 1450px; }}

/* white rounded cards (st.container with a key starting "card") */
div[class*="st-key-card"] {{
  background: #FFFFFF; border-radius: 22px; padding: 20px 22px 10px 22px;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 10px 30px rgba(30,64,175,.07);
}}

.ph, .ph * {{ font-family: '{FONT}', sans-serif; }}
.ph-top {{ display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 18px; }}
.ph-title {{ font-size: 2.1rem; font-weight: 800; color: {INK}; letter-spacing: -.02em; line-height: 1.1; }}
.ph-sub {{ color: {MUTED}; font-size: .95rem; margin-top: 4px; }}
.ph-chip {{ display:inline-flex; align-items:center; gap:8px; background:#FFFFFF; color:{INK};
  border-radius: 999px; padding: 8px 16px; font-weight: 600; font-size: .9rem;
  box-shadow: 0 1px 2px rgba(15,23,42,.05), 0 6px 16px rgba(30,64,175,.06); }}
.dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; }}

/* hero status card */
.hero {{ border-radius: 24px; padding: 24px 28px 22px 28px; color: #FFFFFF; position: relative; overflow: hidden;
  box-shadow: 0 14px 34px rgba(29,78,216,.28); min-height: 300px; }}
.hero.ok  {{ background: linear-gradient(135deg, {BLUE_900} 0%, {BLUE_700} 50%, #60A5FA 100%); }}
.hero.bad {{ background: linear-gradient(135deg, #7F1D1D 0%, {RED} 55%, #F87171 100%);
  box-shadow: 0 14px 34px rgba(220,38,38,.30); }}
.hero-row {{ display:flex; justify-content:space-between; align-items:center; }}
.hero-label {{ font-size: .95rem; font-weight: 600; opacity: .9; }}
.hero-phase {{ background: rgba(255,255,255,.18); border-radius: 999px; padding: 5px 12px; font-size: .82rem; font-weight: 600; }}
.hero-status {{ font-size: 3.2rem; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; margin-top: 10px; }}
.hero-src {{ font-size: .85rem; opacity: .8; margin-top: 2px; }}
.hero svg {{ display:block; width:100%; height:70px; margin: 10px 0 8px 0; }}
.hero-stats {{ display:flex; gap: 0; border-top: 1px solid rgba(255,255,255,.22); padding-top: 14px; }}
.hero-stat {{ flex:1; padding-right: 12px; }}
.hero-stat + .hero-stat {{ border-left: 1px solid rgba(255,255,255,.22); padding-left: 16px; }}
.hero-stat .k {{ font-size: .8rem; opacity: .8; }}
.hero-stat .v {{ font-size: 1.25rem; font-weight: 700; margin-top: 2px; }}

/* small tiles on the right */
.tile {{ background:#FFFFFF; border-radius: 20px; padding: 16px 18px; margin-bottom: 14px; display:flex; gap: 14px; align-items:center;
  box-shadow: 0 1px 2px rgba(15,23,42,.04), 0 10px 30px rgba(30,64,175,.07); }}
.tile:last-child {{ margin-bottom: 0; }}
.tile-icon {{ width: 52px; height: 52px; border-radius: 14px; background: {BLUE_100}; color: {BLUE_700};
  display:flex; align-items:center; justify-content:center; font-weight: 800; font-size: 1.05rem; flex: none; }}
.tile-icon.bad {{ background: #FEE2E2; color: {RED}; }}
.tile-body {{ flex: 1; min-width: 0; }}
.tile-k {{ color: {MUTED}; font-size: .82rem; font-weight: 600; }}
.tile-v {{ color: {INK}; font-size: 1.35rem; font-weight: 800; }}
.bar {{ height: 7px; background: #E8EEF9; border-radius: 999px; margin-top: 8px; overflow: hidden; }}
.bar > span {{ display:block; height:100%; border-radius: 999px; background: linear-gradient(90deg, {BLUE_700}, {BLUE_500}); }}

/* subsystem card header */
.sub-head {{ display:flex; justify-content:space-between; align-items:center; }}
.sub-title {{ font-weight: 700; color: {INK}; font-size: 1.05rem; }}
.sub-val {{ font-size: 2rem; font-weight: 800; color: {INK}; letter-spacing: -.02em; margin-top: 6px; }}
.sub-val small {{ font-size: .95rem; font-weight: 600; color: {MUTED}; margin-left: 4px; }}
.sub-note {{ color: {MUTED}; font-size: .82rem; margin-top: 2px; }}
.pill {{ border-radius: 999px; padding: 4px 12px; font-size: .75rem; font-weight: 700; letter-spacing: .04em; }}
.pill.ok {{ background: {BLUE_100}; color: {BLUE_700}; }}
.pill.bad {{ background: #FEE2E2; color: #B91C1C; }}
.pill.na {{ background: #E2E8F0; color: {MUTED}; }}
.offline {{ height: 190px; display:flex; align-items:center; justify-content:center; text-align:center;
  color: {MUTED}; font-size: .9rem; background: #F5F8FD; border-radius: 14px; margin: 12px 0 14px 0; padding: 0 20px; }}

/* phone alerts banner */
div[class*="st-key-card_ntfy"] {{ border: 1.5px solid {BLUE_100}; }}
.nt-title {{ font-weight: 800; color: {INK}; font-size: 1.1rem; }}
.nt-sub {{ color: {MUTED}; font-size: .9rem; margin: 2px 0 10px 0; }}
.nt-steps {{ color: {INK}; font-size: .9rem; line-height: 1.7; margin: 0; padding-left: 1.2rem; }}
.nt-topic {{ font-family: ui-monospace, Menlo, monospace !important; background: {BLUE_100}; color: {BLUE_700};
  border-radius: 6px; padding: 1px 6px; font-size: .85rem; white-space: nowrap; }}
.qr {{ width: 140px; height: 140px; background: #FFFFFF; border-radius: 10px; }}
.qr path {{ fill: {INK}; }}

/* alerts list */
.al-head {{ display:flex; justify-content:space-between; align-items:center; margin-bottom: 6px; }}
.al-title {{ font-weight: 700; color: {INK}; font-size: 1.05rem; }}
.al-row {{ display:flex; align-items:center; gap: 14px; padding: 12px 4px; border-top: 1px solid #EEF2F9; }}
.al-icon {{ width: 36px; height: 36px; border-radius: 10px; display:flex; align-items:center; justify-content:center; font-weight: 800; flex: none; }}
.al-icon.bad {{ background: #FEE2E2; color: {RED}; }}
.al-icon.ok {{ background: {BLUE_100}; color: {BLUE_700}; }}
.al-main {{ flex: 1; }}
.al-main b {{ color: {INK}; font-weight: 700; }}
.al-main div {{ color: {MUTED}; font-size: .8rem; }}
.al-time {{ color: {MUTED}; font-size: .85rem; font-variant-numeric: tabular-nums; }}
.al-empty {{ color: {MUTED}; font-size: .9rem; padding: 14px 4px 18px 4px; border-top: 1px solid #EEF2F9; }}
</style>
""", unsafe_allow_html=True)


def html(s: str):
    """Render custom HTML. Lines are stripped so Markdown never turns indented HTML into a code block."""
    st.markdown("\n".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Turning raw logger samples into one row per second
# ---------------------------------------------------------------------------
def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw print_logger.py samples -> one row per second (same idea as its Summary sheet).

    Columns out: time_s, rms_x, rms_y, rms_z, vib_rms (m/s^2, gravity removed),
    filament_total_mm, filament_speed_mm_s, temp_c.
    """
    suf = "m_s2" if "accel_x_m_s2" in raw.columns else "g"
    scale = 1.0 if suf == "m_s2" else G          # always show m/s^2
    raw = raw.dropna(subset=["time_s"])
    sec = np.floor(raw["time_s"]).astype(int)
    out = pd.DataFrame(index=pd.Index(sorted(sec.unique()), name="time_s"))
    for a in "xyz":
        # std inside each second = shaking with gravity and slow tilt removed
        out[f"rms_{a}"] = (raw[f"accel_{a}_{suf}"] * scale).groupby(sec).std(ddof=0)
    out["vib_rms"] = np.sqrt(out.rms_x ** 2 + out.rms_y ** 2 + out.rms_z ** 2)
    fil = raw["filament_mm"].groupby(sec).last()
    out["filament_total_mm"] = fil
    out["filament_speed_mm_s"] = fil.diff().fillna(0.0)
    temp = pd.to_numeric(raw.get("hotend_temp_C"), errors="coerce")
    out["temp_c"] = temp.groupby(sec).mean() if temp is not None else np.nan
    return out.reset_index()


def tail_csv(path: str, seconds: float) -> pd.DataFrame:
    """Read only the last few seconds of a (possibly huge, still growing) logger CSV."""
    with open(path, "rb") as f:
        header = f.readline()
        f.seek(0, os.SEEK_END)
        size = f.tell()
        start = max(len(header), size - int(seconds * APPROX_RATE * BYTES_PER_ROW))
        f.seek(start)
        data = f.read()
    if start > len(header):                      # landed mid-line: drop the partial first line
        data = data.split(b"\n", 1)[-1]
    if not data.endswith(b"\n"):                 # logger may be mid-write: drop the partial last line
        data = data.rsplit(b"\n", 1)[0] + b"\n"
    return pd.read_csv(io.BytesIO(header + data))


@st.cache_data(show_spinner="Summarizing recording...")
def load_replay(file_bytes: bytes) -> pd.DataFrame:
    cols = lambda c: c in ("time_s", "filament_mm", "hotend_temp_C") or c.startswith("accel_") and not c.endswith("counts")
    raw = pd.read_csv(io.BytesIO(file_bytes), usecols=cols)
    return summarize(raw)


# ---------------------------------------------------------------------------
# Simulated data (healthy printing, plus injectable faults)
# ---------------------------------------------------------------------------
FAULTS = {
    "none": "None (healthy)",
    "clog": "Feeder: clog / jam (filament stops)",
    "vibration": "Movement system: loose belt / motor (vibration jumps)",
    "heater": "Nozzle: heater fault (temperature drops)",
}


def sim_step(state: dict) -> dict:
    """Make one new per-second row of fake data."""
    rng = np.random.default_rng()
    t = state["t"]
    printing = t >= 30                            # 30 s idle baseline, like the recording rules
    fault = state["fault"] if printing else "none"

    if printing:
        rms = np.abs(rng.normal([0.45, 0.40, 0.25], 0.06))
        speed = max(0.0, rng.normal(1.6, 0.35))
        if rng.random() < 0.05:                   # occasional retraction
            speed = -rng.uniform(0.5, 1.5)
        target = 210.0
    else:
        rms = np.abs(rng.normal([0.04, 0.04, 0.03], 0.01))
        speed = 0.0
        target = 210.0

    secs_in = t - state["fault_start"] if state["fault_start"] is not None else 0
    if fault == "clog":
        speed = abs(rng.normal(0, 0.03))
    elif fault == "vibration":
        rms = rms * (2.5 + 0.02 * secs_in)
    elif fault == "heater":
        target = max(170.0, 210.0 - 1.5 * secs_in)

    state["temp"] += 0.3 * (target - state["temp"]) + rng.normal(0, 0.25)
    state["fil"] += speed
    row = {
        "time_s": t,
        "rms_x": rms[0], "rms_y": rms[1], "rms_z": rms[2],
        "vib_rms": float(np.sqrt((rms ** 2).sum())),
        "filament_total_mm": state["fil"],
        "filament_speed_mm_s": speed,
        "temp_c": state["temp"],
    }
    state["t"] += 1
    return row


def sim_state() -> dict:
    if "sim" not in st.session_state:
        st.session_state.sim = {"t": 0, "fil": 0.0, "temp": 210.0, "fault": "none",
                                "fault_start": None, "rows": []}
    return st.session_state.sim


# ---------------------------------------------------------------------------
# Model: HEALTHY / FAILING
# ---------------------------------------------------------------------------
def predict(features: pd.DataFrame, cfg: dict) -> dict:
    """Decide HEALTHY or FAILING from the last few seconds of per-second features.

    Input:  features, a DataFrame with the summarize() columns, newest row last.
    Output: {"label": "HEALTHY" | "FAILING",
             "confidence": 0..1 or None,
             "subsystems": {"Nozzle": bool | None, "Feeder": bool, "Movement system": bool},
             "source": short text shown under the status}
            subsystems maps each subsystem to True if it looks like it is failing
            (None = no data for it).

    PLACEHOLDER: threshold rules. Bella / Austin: replace this body with the real model
    (e.g. model.predict on features built from the same columns) and return the same dict.
    """
    recent = features.tail(cfg["persist_s"])
    if len(recent) < cfg["persist_s"]:
        return {"label": "HEALTHY", "confidence": None,
                "subsystems": {"Nozzle": None, "Feeder": False, "Movement system": False},
                "source": "warming up"}

    printing = recent["vib_rms"] > cfg["printing_vib"]
    vib_bad = bool((recent["vib_rms"] > cfg["vib_limit"]).all())
    flow_bad = bool(printing.all() and (recent["filament_speed_mm_s"].abs() < cfg["min_speed"]).all())
    temps = recent["temp_c"].dropna()
    if len(temps) < len(recent):
        heat_bad = None                           # thermocouple off / missing
    else:
        heat_bad = bool(printing.all() and ((temps - cfg["target_temp"]).abs() > cfg["temp_band"]).all())

    subs = {"Nozzle": heat_bad, "Feeder": flow_bad, "Movement system": vib_bad}
    failing = any(v for v in subs.values())
    return {"label": "FAILING" if failing else "HEALTHY", "confidence": None,
            "subsystems": subs, "source": "placeholder rules (ML model not connected)"}


def update_alerts(result: dict, t: float):
    """Log an alert the moment a subsystem flips to failing, and when it recovers."""
    log = st.session_state.setdefault("alerts", [])
    prev = st.session_state.setdefault("prev_subs", {})
    for name, bad in result["subsystems"].items():
        if bad is None:
            continue
        if bad and not prev.get(name):
            log.insert(0, {"time_s": int(t), "clock": time.strftime("%H:%M:%S"),
                           "subsystem": name, "event": "FAILING"})
        elif not bad and prev.get(name):
            log.insert(0, {"time_s": int(t), "clock": time.strftime("%H:%M:%S"),
                           "subsystem": name, "event": "back to healthy"})
        prev[name] = bad


NOTIFY_COOLDOWN_S = 30          # at most one phone alert per 30 s; the latest status is sent after
REPLAY_SOURCES = {"API log": "Replay", "Replay CSV": "Replay", "Simulated": "Simulated"}


def notify_status(result: dict, details: dict, job: str = ""):
    """Push a phone alert when the overall status flips between HEALTHY and FAILING.

    details maps subsystem name -> one line describing what is wrong with it right now.
    Sends the current status whenever it differs from the last one sent, at most once per
    cooldown, so a status that flaps doesn't spam phones and the phone never ends up stale.
    """
    if not alerts_on or result.get("source") == "warming up":
        return
    label = result["label"]
    sent = st.session_state.setdefault("ntfy_sent", {"label": "HEALTHY", "t": 0.0, "since": None})
    if label == sent["label"] or time.time() - sent["t"] < NOTIFY_COOLDOWN_S:
        return

    stamp = notify.stamp()
    prefix = f"[{REPLAY_SOURCES[source]}] " if source in REPLAY_SOURCES else ""
    failing = [k for k, v in result["subsystems"].items() if v]
    if label == "FAILING":
        title = f"{prefix}Printer FAILING: {', '.join(failing)}"
        lines = [details.get(k, f"{k} looks wrong") for k in failing]
        prio, tags = 5, ["rotating_light"]
        sent["since"] = time.time()
    else:
        title = f"{prefix}Printer back to HEALTHY"
        mins = (time.time() - sent["since"]) / 60 if sent["since"] else None
        lines = ["All subsystems look normal again."]
        if mins is not None:
            lines.append(f"It was failing for about {max(1, round(mins))} min.")
        prio, tags = 3, ["white_check_mark"]
        sent["since"] = None
    lines.append(stamp + (f" · {job}" if job else ""))
    notify.send_async(NTFY_SERVER, st.session_state.ntfy_topic, title, "\n".join(lines), ntfy_status, prio, tags)
    sent.update(label=label, t=time.time())


def api_predict(df: pd.DataFrame, cfg: dict) -> dict:
    """HEALTHY / FAILING from printer API rows (printer_api.to_frame columns, newest last).

    Same output dict as predict(). Subsystems: Nozzle, Bed, Motion.
    PLACEHOLDER: threshold rules using the printer's own temperature targets. Swap the body
    for the real model once it is trained on API features (same column names as
    data/preprocessed_api.py), and keep the returned dict the same.
    """
    n = int(cfg["api_persist_n"])
    recent = df.tail(n)
    if len(recent) < n:
        return {"label": "HEALTHY", "confidence": None,
                "subsystems": {"Nozzle": None, "Bed": None, "Motion": None}, "source": "warming up"}

    printing = bool(recent["printing"].all())

    def temp_bad(err_col, target_col, band):
        if recent[err_col].isna().all():
            return None
        heated = recent[target_col] > 0          # target 0 = heater off on purpose, not a fault
        return bool(printing and heated.all() and (recent[err_col].abs() > band).all())

    nozzle = temp_bad("nozzle_error_c", "extruder1_temp_target", cfg["api_nozzle_band"])
    bed = temp_bad("bed_error_c", "bed_temp_target", cfg["api_bed_band"])
    speed = recent["head_speed_mm_s"]
    motion = None if speed.isna().all() else bool(printing and (speed.fillna(0) < cfg["api_stall_speed"]).all())
    if recent["printer_status"].iloc[-1] == "error":
        motion = True

    subs = {"Nozzle": nozzle, "Bed": bed, "Motion": motion}
    failing = any(v for v in subs.values())
    return {"label": "FAILING" if failing else "HEALTHY", "confidence": None,
            "subsystems": subs, "source": "placeholder rules on printer API data (ML model not connected)"}


@st.cache_data(show_spinner="Reading API log...")
def load_api_log(file_bytes: bytes) -> pd.DataFrame:
    return printer_api.to_frame(printer_api.parse_log_text(file_bytes.decode("utf-8", errors="replace")))


@st.cache_data(show_spinner=False, max_entries=4)
def load_api_log_path(path: str, mtime: float, size: int) -> pd.DataFrame:
    """Parse a log on disk. mtime/size are only cache keys: the file is re-read when it grows."""
    with open(path, encoding="utf-8", errors="replace") as f:
        return printer_api.to_frame(printer_api.parse_log_text(f.read()))


def newest_api_log(path: str) -> str | None:
    """A file path is used as is; a folder means the newest api_data*.txt inside it."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        logs = [os.path.join(path, f) for f in os.listdir(path) if f.startswith("api_data") and f.endswith(".txt")]
        return max(logs, key=os.path.getmtime) if logs else None
    return None


API_SOURCES = ("Printer API", "Live API log", "API log")
API_RETRY_S = 5                 # seconds to wait before retrying an unreachable printer


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("Printer Health")
source = st.sidebar.radio("Data source", ["Printer API", "Live API log", "API log", "Simulated", "Replay CSV", "Live CSV"],
                          captions=["dashboard polls the printer", "follows Bella's data/api.py", "replay a data/api.py log",
                                    "sensors, fake data", "sensors, recording", "sensors, print_logger.py"])
window_s = st.sidebar.slider("Chart window (seconds)", 30, 600, 120, step=30)

if source == "Printer API":
    api_cfg = printer_api.get_config(st.secrets if os.path.exists(".streamlit/secrets.toml") else None)
    api_cfg["url"] = st.sidebar.text_input("Printer address", api_cfg["url"]).rstrip("/")
    if not api_cfg["user"]:
        st.sidebar.caption("No API credentials found. Reads usually work without them; "
                           "if the printer answers 401, add them to .streamlit/secrets.toml.")
    if st.sidebar.button("Clear history"):
        for k in ("api_rows", "hist", "alerts", "prev_subs", "api_retry_at"):
            st.session_state.pop(k, None)
elif source == "Live API log":
    api_log_path = st.sidebar.text_input("Folder data/api.py writes to (or one log file)",
                                         placeholder="e.g. ~/Predictive-Maintenance-Problem/data/api_data")
    st.sidebar.caption("Uses the newest api_data_*.txt in the folder, so starting a new collector run "
                       "switches the dashboard to it automatically.")
elif source == "API log":
    api_up = st.sidebar.file_uploader("Log from data/api.py (api_data_*.txt)", type="txt")
    speed_x = st.sidebar.select_slider("Playback speed (samples per refresh)", [1, 2, 5, 10, 30], value=5)
    if api_up is not None and st.session_state.get("api_replay_name") != api_up.name:
        st.session_state.api_replay_name = api_up.name
        st.session_state.api_replay_pos = 0
        for k in ("hist", "alerts", "prev_subs"):
            st.session_state.pop(k, None)
elif source == "Simulated":
    sim = sim_state()
    choice = st.sidebar.selectbox("Inject fault", list(FAULTS), format_func=FAULTS.get,
                                  index=list(FAULTS).index(sim["fault"]))
    if choice != sim["fault"]:
        sim["fault"] = choice
        sim["fault_start"] = sim["t"] if choice != "none" else None
    if st.sidebar.button("Restart simulation"):
        del st.session_state["sim"]
        st.session_state.pop("alerts", None)
        st.session_state.pop("prev_subs", None)
        st.session_state.pop("hist", None)
        st.rerun()
    st.sidebar.caption("First 30 s is idle, then printing starts. Faults take effect once printing.")
elif source == "Replay CSV":
    up = st.sidebar.file_uploader("Recording from print_logger.py (.csv)", type="csv")
    speed_x = st.sidebar.select_slider("Playback speed", [1, 2, 5, 10, 30, 60], value=10)
    if up is not None and st.session_state.get("replay_name") != up.name:
        st.session_state.replay_name = up.name
        st.session_state.replay_pos = 0
        st.session_state.pop("alerts", None)
        st.session_state.pop("prev_subs", None)
        st.session_state.pop("hist", None)
else:
    live_path = st.sidebar.text_input("Path to the CSV being recorded",
                                      placeholder="e.g. ~/sensor_logs/print_20261103_190000.csv")

with st.sidebar.expander("Placeholder rule settings"):
    if source in API_SOURCES:
        cfg = {
            "api_nozzle_band": st.number_input("Nozzle: allowed deviation from target (°C)", value=10.0, step=1.0),
            "api_bed_band": st.number_input("Bed: allowed deviation from target (°C)", value=5.0, step=1.0),
            "api_stall_speed": st.number_input("Motion: stalled if head slower than (mm/s)", value=0.5, step=0.1),
            "api_persist_n": st.number_input("Must last this many samples to count", value=8, min_value=1, step=1),
        }
    else:
        cfg = {
            "vib_limit": st.number_input("Vibration limit (m/s², combined RMS)", value=1.5, step=0.1),
            "printing_vib": st.number_input("Printing if vibration above (m/s²)", value=0.3, step=0.05),
            "min_speed": st.number_input("Clog if filament speed below (mm/s)", value=0.2, step=0.05),
            "target_temp": st.number_input("Nozzle target (°C)", value=210.0, step=5.0),
            "temp_band": st.number_input("Allowed temp deviation (°C)", value=10.0, step=1.0),
            "persist_s": st.number_input("Must last this many seconds to count", value=5, min_value=1, step=1),
        }
    st.caption("These only matter until the ML model replaces predict().")

_secrets = st.secrets if os.path.exists(".streamlit/secrets.toml") else None
NTFY_SERVER = notify.server(_secrets)
if "ntfy_topic" not in st.session_state:
    st.session_state.ntfy_topic = notify.load_topic(_secrets)
ntfy_status = st.session_state.setdefault("ntfy_status", {"ok": None, "error": None})

with st.sidebar.expander("Phone alerts"):
    alerts_on = st.toggle("Send phone alerts", value=True,
                          help="Push notification to every phone subscribed to the topic when the status "
                               "changes between HEALTHY and FAILING.")
    new_topic = st.text_input("ntfy topic", st.session_state.ntfy_topic,
                              help="Use the same topic on every laptop that runs the dashboard so the "
                                   "team gets alerts from all of them.")
    if new_topic.strip() and new_topic.strip() != st.session_state.ntfy_topic:
        st.session_state.ntfy_topic = new_topic.strip()
        notify.save_topic(st.session_state.ntfy_topic)
    if st.button("Show phone setup"):
        st.session_state.ntfy_banner_hidden = False
    if ntfy_status["error"]:
        st.caption(f"Last alert failed to send: {ntfy_status['error'][:120]}")


# ---------------------------------------------------------------------------
# Main view (refreshes every second)
# ---------------------------------------------------------------------------
def get_features():
    """Return (per-second DataFrame up to now, message or None)."""
    if source == "Simulated":
        sim = sim_state()
        sim["rows"].append(sim_step(sim))
        sim["rows"] = sim["rows"][-1200:]
        return pd.DataFrame(sim["rows"]), None
    if source == "Replay CSV":
        if up is None:
            return None, "Upload a CSV from print_logger.py in the sidebar."
        df = load_replay(up.getvalue())
        pos = st.session_state.get("replay_pos", 0) + speed_x
        st.session_state.replay_pos = min(pos, len(df))
        if st.session_state.replay_pos >= len(df):
            st.toast("End of recording")
        return df.iloc[: st.session_state.replay_pos], None
    path = os.path.expanduser(live_path or "")
    if not path:
        return None, "Enter the path of the CSV that print_logger.py is writing."
    if not os.path.exists(path):
        return None, f"File not found: {path}"
    try:
        raw = tail_csv(path, window_s + 2)
    except Exception as e:                        # e.g. logger just started, file still empty
        return None, f"Waiting for data ({e})"
    if raw.empty:
        return None, "Waiting for data..."
    df = summarize(raw)
    return df.iloc[:-1] if len(df) > 1 else df, None   # last second is usually incomplete


def get_api_features():
    """Return (printer API DataFrame up to now, message or None)."""
    if source == "API log":
        if api_up is None:
            return None, "Upload an api_data_*.txt log (written by data/api.py) in the sidebar."
        df = load_api_log(api_up.getvalue())
        if df.empty:
            return None, "No /api/v1/printer snapshots found in that file."
        pos = min(st.session_state.get("api_replay_pos", 0) + speed_x, len(df))
        st.session_state.api_replay_pos = pos
        if pos >= len(df):
            st.toast("End of log")
        return df.iloc[:pos], None

    if source == "Live API log":
        folder = os.path.expanduser(api_log_path or "")
        if not folder:
            return None, "Enter the folder Bella's data/api.py writes to (its data/api_data folder)."
        log = newest_api_log(folder)
        if log is None:
            return None, f"No api_data_*.txt found at {folder}. Is data/api.py running?"
        st_ = os.stat(log)
        df = load_api_log_path(log, st_.st_mtime, st_.st_size)
        if df.empty:
            return None, f"Waiting for the first printer snapshot in {os.path.basename(log)}..."
        if time.time() - st_.st_mtime > 15:
            st.session_state.api_error = f"{os.path.basename(log)} hasn't changed in {time.time() - st_.st_mtime:.0f} s. Is data/api.py still running?"
        else:
            st.session_state.api_error = None
        return df, None

    rows = st.session_state.setdefault("api_rows", [])
    # After a failed poll, wait before trying again. A poll that times out takes longer than
    # the 1 s refresh, so retrying every refresh would keep the page from ever drawing.
    retry_at = st.session_state.get("api_retry_at", 0.0)
    if time.time() >= retry_at:
        try:
            row = printer_api.poll_once(api_cfg)
            row["t_wall"] = time.time()
            rows.append(row)
            del rows[:-2000]                      # keep about the last 50 minutes
            st.session_state.api_error = None
            st.session_state.api_retry_at = 0.0
        except requests.RequestException as e:
            st.session_state.api_error = str(e)
            st.session_state.api_retry_at = time.time() + API_RETRY_S
    if not rows:
        wait = max(0, int(st.session_state.get("api_retry_at", 0) - time.time()))
        return None, (f"Can't reach the printer. Make sure the printer is turned on and you are connected to "
                      f"eduroam. Trying again in {wait} s. To look at recorded data instead, pick API log in "
                      f"the sidebar.")
    return printer_api.to_frame(rows), None


def sparkline(values, w=600, h=70, min_range=None) -> str:
    """Small white trend line for the hero card, as inline SVG."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return ""
    lo, hi = v.min(), v.max()
    if min_range and hi - lo < min_range:          # keep small noise looking small
        mid = (hi + lo) / 2
        lo, hi = mid - min_range / 2, mid + min_range / 2
    rng = hi - lo if hi > lo else 1.0
    xs = np.linspace(0, w, len(v))
    ys = h - 6 - (v - lo) / rng * (h - 12)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    area = f"0,{h} {pts} {w},{h}"
    return (f"<svg viewBox='0 0 {w} {h}' preserveAspectRatio='none'>"
            f"<polygon points='{area}' fill='rgba(255,255,255,.12)'/>"
            f"<polyline points='{pts}' fill='none' stroke='white' stroke-width='2.2' "
            f"stroke-linejoin='round' vector-effect='non-scaling-stroke'/></svg>")


def chart(win: pd.DataFrame, cols: dict, colors: list, rule=None, zero=True, area=True, domain=None):
    """Blue Altair chart for a subsystem card. cols maps column name -> legend label."""
    d = win.reset_index()[["time_s", *cols]].rename(columns=cols)
    d = d.melt("time_s", list(cols.values()), "series", "value")
    multi = len(cols) > 1
    x = alt.X("time_s:Q", title=None, axis=alt.Axis(grid=False, tickCount=5, format="d"))
    y = alt.Y("value:Q", title=None, scale=alt.Scale(zero=zero, nice=True, domain=domain) if domain else alt.Scale(zero=zero, nice=True), axis=alt.Axis(tickCount=4))
    if multi:
        layer = alt.Chart(d).mark_line(strokeWidth=2, interpolate="monotone").encode(
            x=x, y=y, color=alt.Color("series:N", scale=alt.Scale(domain=list(cols.values()), range=colors),
                                      legend=alt.Legend(orient="bottom", title=None, symbolType="circle")))
    elif area:
        grad = alt.Gradient(gradient="linear", x1=1, x2=1, y1=1, y2=0,
                            stops=[alt.GradientStop(color="rgba(59,130,246,0.02)", offset=0),
                                   alt.GradientStop(color="rgba(59,130,246,0.35)", offset=1)])
        layer = alt.Chart(d).mark_area(line={"color": colors[0], "strokeWidth": 2}, color=grad,
                                       interpolate="monotone").encode(x=x, y=y)
    else:
        layer = alt.Chart(d).mark_line(color=colors[0], strokeWidth=2.2, interpolate="monotone",
                                       clip=True).encode(x=x, y=y)
    if rule is not None:
        layer = layer + alt.Chart(pd.DataFrame({"v": [rule]})).mark_rule(
            color="#F87171", strokeDash=[5, 4], strokeWidth=1.5).encode(y="v:Q")
    return (layer.properties(height=190)
            .configure(background="transparent", font=f"{FONT}, -apple-system, Helvetica, Arial, sans-serif", padding={"left": 2, "right": 10, "top": 4, "bottom": 2})
            .configure_view(strokeWidth=0)
            .configure_axis(gridColor="#EEF2F9", domain=False, ticks=False,
                            labelColor="#94A3B8", labelFontSize=11, labelPadding=6)
            .configure_legend(labelColor=MUTED, labelFontSize=11))


def pill(v) -> str:
    if v is None:
        return "<span class='pill na'>NO DATA</span>"
    return "<span class='pill bad'>FAILING</span>" if v else "<span class='pill ok'>HEALTHY</span>"


SOURCE_LABEL = {"Simulated": "Simulated data", "Replay CSV": "Replaying recording", "Live CSV": "Live from sensors",
                "Printer API": "Live from printer API", "Live API log": "Live from api.py log",
                "API log": "Replaying API log"}


def render_alerts(alerts: list, n_fail: int):
    rows = "".join(
        f"""<div class="al-row"><div class="al-icon {'bad' if a['event'] == 'FAILING' else 'ok'}">{'!' if a['event'] == 'FAILING' else '✓'}</div>
        <div class="al-main"><b>{a['subsystem']}</b> {'started failing' if a['event'] == 'FAILING' else 'back to healthy'}
        <div>at {a['time_s'] // 60}:{a['time_s'] % 60:02d} into the session</div></div>
        <div class="al-time">{a['clock']}</div></div>""" for a in alerts[:8])
    with st.container(key="card_alerts"):
        html(f"""<div class="ph"><div class="al-head"><span class="al-title">Alerts</span>
             <span class="pill {'bad' if n_fail else 'ok'}">{n_fail} FAILING EVENT{'S' if n_fail != 1 else ''}</span></div>
             {rows or "<div class='al-empty'>No alerts yet. Everything has stayed healthy.</div>"}</div>""")


def fmt_secs(x) -> str:
    if x is None or pd.isna(x):
        return "--"
    m, s = divmod(int(x), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def temp_card(win, last, title, sub, cur, tgt, band, key):
    """Subsystem card for a heater: current temperature line plus the printer's own target."""
    with st.container(key=key):
        target_now = last[tgt]
        note = f"Target {target_now:.0f} °C (dashed) · from printer API" if target_now > 0 else "Heater off (target 0 °C)"
        html(f"""<div class="ph"><div class="sub-head"><span class="sub-title">{title}</span>{pill(sub)}</div>
             <div class="sub-val">{last[cur]:.1f}<small>°C</small></div>
             <div class="sub-note">{note}</div></div>""")
        vals = pd.concat([win[cur], win[tgt][win[tgt] > 0]]).dropna()
        lo, hi = (vals.min() - band, vals.max() + band) if len(vals) else (0, 1)
        if (win[tgt] > 0).any():
            d = win[[cur, tgt]].rename(columns={cur: "Current", tgt: "Target"})
            d["Target"] = d["Target"].where(d["Target"] > 0)      # heater off: no target line
            st.altair_chart(chart(d, {"Current": "Current", "Target": "Target"}, [BLUE_700, "#F87171"],
                                  zero=False, domain=[lo, hi]), width="stretch")
        else:
            st.altair_chart(chart(win, {cur: "Current"}, [BLUE_700], zero=False, area=False,
                                  domain=[lo, hi]), width="stretch")


def api_view():
    df, msg = get_api_features()
    if msg or df is None or df.empty:
        header(None)
        st.info(msg or "Waiting for data...")
        return
    if st.session_state.get("api_error") and source == "Printer API":
        st.warning("Lost connection to the printer, showing the last data received. Trying again...")
    elif st.session_state.get("api_error") and source == "Live API log":
        st.warning(st.session_state.api_error)

    result = api_predict(df, cfg)
    t_now = df["time_s"].iloc[-1]
    update_alerts(result, t_now)
    failing = result["label"] == "FAILING"
    hist = st.session_state.setdefault("hist", {})
    hist[len(df)] = failing
    last = df.iloc[-1]
    notify_status(result, {
        "Nozzle": f"Nozzle {last['extruder1_temp_current']:.1f} °C, target {last['extruder1_temp_target']:.0f} °C "
                  f"({last['nozzle_error_c']:+.1f} °C off)",
        "Bed": f"Bed {last['bed_temp_current']:.1f} °C, target {last['bed_temp_target']:.0f} °C "
               f"({last['bed_error_c']:+.1f} °C off)",
        "Motion": ("Printer reported an error" if last["printer_status"] == "error" else
                   f"Print head looks stalled ({last['head_speed_mm_s']:.1f} mm/s) while printing"),
    }, job=last["job_name"])
    win = df[df["time_s"] > t_now - window_s].set_index("time_s")
    printing = bool(last["printing"])
    alerts = st.session_state.get("alerts", [])
    n_fail = sum(1 for a in alerts if a["event"] == "FAILING")
    subs = result["subsystems"]

    header(printing)

    # --- Row 1: hero status card + tiles ---
    left, right = st.columns([2.3, 1], gap="medium")
    with left:
        conf = result["confidence"]
        src = result["source"] + (f" · confidence {conf:.0%}" if conf is not None else "")
        job = last["job_name"] or "No print job"
        phase = f"● {last['job_state'].replace('_', ' ').capitalize()}" if last["job_state"] else f"○ {last['printer_status'].capitalize() or 'Idle'}"
        speed = last["head_speed_mm_s"]
        html(f"""
        <div class="ph hero {'bad' if failing else 'ok'}">
          <div class="hero-row"><span class="hero-label">Printer status · {job}</span>
            <span class="hero-phase">{phase}</span></div>
          <div class="hero-status">{result['label']}</div>
          <div class="hero-src">{src}</div>
          {sparkline(win['extruder1_temp_current'].values, min_range=10)}
          <div class="hero-stats">
            <div class="hero-stat"><div class="k">Nozzle</div><div class="v">{last['extruder1_temp_current']:.1f} / {last['extruder1_temp_target']:.0f} °C</div></div>
            <div class="hero-stat"><div class="k">Bed</div><div class="v">{last['bed_temp_current']:.1f} / {last['bed_temp_target']:.0f} °C</div></div>
            <div class="hero-stat"><div class="k">Head speed</div><div class="v">{'--' if pd.isna(speed) else f'{speed:.1f} mm/s'}</div></div>
          </div>
        </div>""")
    with right:
        prog = last["job_progress"]
        prog_pct = 0.0 if pd.isna(prog) else 100 * prog
        remaining = last["job_time_total"] - last["job_time_elapsed"]
        pct = 100 * (1 - sum(hist.values()) / max(len(hist), 1))
        html(f"""
        <div class="ph">
          <div class="tile"><div class="tile-icon">{prog_pct:.0f}%</div>
            <div class="tile-body"><div class="tile-k">Print progress</div><div class="tile-v">{fmt_secs(remaining)} left</div>
            <div class="bar"><span style="width:{prog_pct:.1f}%"></span></div></div></div>
          <div class="tile"><div class="tile-icon {'bad' if failing else ''}">{'!' if failing else '✓'}</div>
            <div class="tile-body"><div class="tile-k">Time healthy</div><div class="tile-v">{pct:.1f}%</div>
            <div class="bar"><span style="width:{pct:.1f}%"></span></div></div></div>
          <div class="tile"><div class="tile-icon {'bad' if n_fail else ''}">{n_fail}</div>
            <div class="tile-body"><div class="tile-k">Alerts this session</div>
            <div class="tile-v">{'None' if not n_fail else (str(n_fail) + (' alert' if n_fail == 1 else ' alerts'))}</div></div></div>
        </div>""")

    st.write("")

    # --- Row 2: subsystem cards ---
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        temp_card(win, last, "Nozzle", subs.get("Nozzle"), "extruder1_temp_current", "extruder1_temp_target",
                  cfg["api_nozzle_band"], "card_api_nozzle")
    with c2:
        temp_card(win, last, "Bed", subs.get("Bed"), "bed_temp_current", "bed_temp_target",
                  cfg["api_bed_band"], "card_api_bed")
    if last[["head_x_mm", "head_y_mm", "head_z_mm"]].isna().any():
        head_pos = "Head position not reported (printer idle)"
    else:
        head_pos = (f"Head at X {last['head_x_mm']:.0f} · Y {last['head_y_mm']:.0f} · "
                    f"Z {last['head_z_mm']:.1f} mm · avg speed between polls")
    with c3, st.container(key="card_api_motion"):
        html(f"""<div class="ph"><div class="sub-head"><span class="sub-title">Motion</span>{pill(subs.get('Motion'))}</div>
             <div class="sub-val">{'--' if pd.isna(speed) else f'{speed:.1f}'}<small>mm/s</small></div>
             <div class="sub-note">{head_pos}</div></div>""")
        if win["head_speed_mm_s"].notna().any():
            st.altair_chart(chart(win, {"head_speed_mm_s": "Head speed"}, [BLUE_700]), width="stretch")
        else:
            st.caption("Head speed shows up once the print head starts moving.")

    st.write("")
    render_alerts(alerts, n_fail)


@st.fragment(run_every=1.0)
def live_view():
    if source in API_SOURCES:
        api_view()
        return
    df, msg = get_features()
    if msg or df is None or df.empty:
        header(None)
        st.info(msg or "Waiting for data...")
        return

    result = predict(df, cfg)
    t_now = df["time_s"].iloc[-1]
    update_alerts(result, t_now)
    failing = result["label"] == "FAILING"
    hist = st.session_state.setdefault("hist", {})
    hist[int(t_now)] = failing
    last = df.iloc[-1]
    notify_status(result, {
        "Nozzle": f"Nozzle {last['temp_c']:.1f} °C, target {cfg['target_temp']:.0f} °C "
                  f"({last['temp_c'] - cfg['target_temp']:+.1f} °C off)",
        "Feeder": f"Filament not moving ({last['filament_speed_mm_s']:.2f} mm/s) while printing. Possible clog",
        "Movement system": f"Vibration {last['vib_rms']:.2f} m/s², above the {cfg['vib_limit']:.1f} m/s² limit",
    })
    win = df[df["time_s"] > t_now - window_s].set_index("time_s")
    printing = last["vib_rms"] > cfg["printing_vib"]
    temp_on = win["temp_c"].notna().any()
    alerts = st.session_state.get("alerts", [])

    header(printing)

    # --- Row 1: hero status card + tiles ---
    left, right = st.columns([2.3, 1], gap="medium")
    with left:
        stat_temp = f"{last['temp_c']:.1f} °C" if temp_on else "offline"
        conf = result["confidence"]
        src = result["source"] + (f" · confidence {conf:.0%}" if conf is not None else "")
        html(f"""
        <div class="ph hero {'bad' if failing else 'ok'}">
          <div class="hero-row"><span class="hero-label">Printer status</span>
            <span class="hero-phase">{'● Printing' if printing else '○ Idle'}</span></div>
          <div class="hero-status">{result['label']}</div>
          <div class="hero-src">{src}</div>
          {sparkline(win['vib_rms'].values)}
          <div class="hero-stats">
            <div class="hero-stat"><div class="k">Movement system</div><div class="v">{last['vib_rms']:.2f} m/s²</div></div>
            <div class="hero-stat"><div class="k">Feeder</div><div class="v">{last['filament_speed_mm_s']:.2f} mm/s</div></div>
            <div class="hero-stat"><div class="k">Nozzle</div><div class="v">{stat_temp}</div></div>
          </div>
        </div>""")
    with right:
        pct = 100 * (1 - sum(hist.values()) / max(len(hist), 1))
        m, s = divmod(int(t_now), 60)
        h, m = divmod(m, 60)
        elapsed = f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
        n_fail = sum(1 for a in alerts if a["event"] == "FAILING")
        html(f"""
        <div class="ph">
          <div class="tile"><div class="tile-icon {'bad' if failing else ''}">{'!' if failing else '✓'}</div>
            <div class="tile-body"><div class="tile-k">Time healthy</div><div class="tile-v">{pct:.1f}%</div>
            <div class="bar"><span style="width:{pct:.1f}%"></span></div></div></div>
          <div class="tile"><div class="tile-icon">⏱</div>
            <div class="tile-body"><div class="tile-k">Elapsed</div><div class="tile-v">{elapsed}</div></div></div>
          <div class="tile"><div class="tile-icon {'bad' if n_fail else ''}">{n_fail}</div>
            <div class="tile-body"><div class="tile-k">Alerts this session</div>
            <div class="tile-v">{'None' if not n_fail else (str(n_fail) + (' alert' if n_fail == 1 else ' alerts'))}</div></div></div>
        </div>""")

    st.write("")

    # --- Row 2: subsystem cards ---
    subs = result["subsystems"]
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1, st.container(key="card_vib"):
        html(f"""<div class="ph"><div class="sub-head"><span class="sub-title">Movement system</span>{pill(subs.get('Movement system'))}</div>
             <div class="sub-val">{last['vib_rms']:.2f}<small>m/s² RMS</small></div>
             <div class="sub-note">Accelerometer · motor + belt · dashed line = alert limit</div></div>""")
        st.altair_chart(chart(win, {"rms_x": "X", "rms_y": "Y", "rms_z": "Z"},
                              [BLUE_900, BLUE_500, BLUE_300], rule=cfg["vib_limit"]), width="stretch")
    with c2, st.container(key="card_flow"):
        html(f"""<div class="ph"><div class="sub-head"><span class="sub-title">Feeder</span>{pill(subs.get('Feeder'))}</div>
             <div class="sub-val">{last['filament_speed_mm_s']:.2f}<small>mm/s</small></div>
             <div class="sub-note">{last['filament_total_mm']:.0f} mm fed · filament motion sensor</div></div>""")
        st.altair_chart(chart(win, {"filament_speed_mm_s": "Feed speed"}, [BLUE_700]), width="stretch")
    with c3, st.container(key="card_temp"):
        if temp_on:
            html(f"""<div class="ph"><div class="sub-head"><span class="sub-title">Nozzle</span>{pill(subs.get('Nozzle'))}</div>
                 <div class="sub-val">{last['temp_c']:.1f}<small>°C</small></div>
                 <div class="sub-note">Target {cfg['target_temp']:.0f} °C (dashed) · thermocouple</div></div>""")
            st.altair_chart(chart(win, {"temp_c": "Temperature"}, [BLUE_700], rule=cfg["target_temp"],
                                  zero=False, area=False,
                                  domain=[min(cfg["target_temp"] - cfg["temp_band"] - 5, win["temp_c"].min()),
                                          max(cfg["target_temp"] + cfg["temp_band"] + 5, win["temp_c"].max())]), width="stretch")
        else:
            html(f"""<div class="ph"><div class="sub-head"><span class="sub-title">Nozzle</span>{pill(None)}</div>
                 <div class="sub-val">off<small></small></div>
                 <div class="sub-note">Thermocouple</div>
                 <div class="offline">Thermocouple is switched off in the firmware<br>(MCP9600 replacement pending)</div></div>""")

    st.write("")

    # --- Row 3: alerts ---
    render_alerts(alerts, n_fail)


def phone_banner():
    """Setup card for phone alerts, under the header until someone hides it."""
    if st.session_state.get("ntfy_banner_hidden") or not alerts_on:
        return
    topic = st.session_state.ntfy_topic
    with st.container(key="card_ntfy"):
        qr_col, text_col = st.columns([1, 4], gap="medium", vertical_alignment="center")
        with qr_col:
            html(notify.qr_svg(notify.subscribe_url(NTFY_SERVER, topic)))
        with text_col:
            html(f"""<div class="ph"><div class="nt-title">Get alerts on your phone</div>
                 <div class="nt-sub">Step away from the dashboard and still hear about it when the printer
                 goes from healthy to failing (or back).</div>
                 <ol class="nt-steps">
                   <li>Install the free <b>ntfy</b> app (App Store or Google Play).</li>
                   <li>Scan the QR code with your camera, or in the app tap <b>+</b> and subscribe to
                       <span class="nt-topic">{topic}</span></li>
                   <li>Send a test alert to check it works.</li>
                 </ol></div>""")
            b1, b2, _ = st.columns([1.5, 0.8, 5], gap="small")
            if b1.button("Send test alert", type="primary"):
                try:
                    notify.send(NTFY_SERVER, topic, "Printer Health test alert",
                                "Phone alerts are working. You'll get a message like this when the printer "
                                "status changes.\n" + notify.stamp(),
                                3, ["bell"])
                    st.toast("Test alert sent. Check your phone.")
                except Exception as e:                  # network blocked, ntfy down, etc.
                    st.toast(f"Couldn't send: {e}")
            if b2.button("Hide"):
                st.session_state.ntfy_banner_hidden = True
                st.rerun()
    st.write("")


def header(printing):
    dot = "#22C55E" if source in ("Live CSV", "Printer API", "Live API log") else BLUE_500
    html(f"""
    <div class="ph ph-top">
      <div><div class="ph-title">Printer Health</div>
        <div class="ph-sub">Predictive maintenance · UltiMaker S3</div></div>
      <div class="ph-chip"><span class="dot" style="background:{dot}"></span>{SOURCE_LABEL[source]}</div>
    </div>""")
    phone_banner()


# history for "Time healthy" starts over when the data source changes
if st.session_state.get("last_source") != source:
    st.session_state.last_source = source
    for k in ("hist", "alerts", "prev_subs", "api_rows", "ntfy_sent"):
        st.session_state.pop(k, None)

live_view()
