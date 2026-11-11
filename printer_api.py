"""
Printer API data for the dashboard (UltiMaker S3 HTTP API).

Two ways to get data, both producing the same DataFrame:
  poll_once()       ask the printer for /api/v1/print_job and /api/v1/printer right now
  parse_log_text()  read a log written by Bella's data/api.py (api_data_*.txt)

The flattened column names match data/preprocessed_api.py on the ML-Models branch
(extruder1_temp_current, bed_temp_target, head_x_mm, ...) so the dashboard and the
model see the same features. The digest-auth code is ported from data/api.py.

Printer address and API credentials come from environment variables or from
.streamlit/secrets.toml (see secrets.toml.example). Keep them out of git.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
from typing import Any

import numpy as np
import pandas as pd
import requests

TIMEOUT_S = (1.5, 3.0)          # (connect, read) seconds

# Same sanity bounds as data/preprocessed_api.py: values outside become NaN.
LIMITS = {
    "extruder1_temp_current": (0.0, 290.0),
    "extruder1_temp_target": (0.0, 290.0),
    "bed_temp_current": (0.0, 150.0),
    "bed_temp_target": (0.0, 150.0),
    "head_x_mm": (0.0, 230.0),
    "head_y_mm": (0.0, 190.0),
    "head_z_mm": (0.0, 200.0),
}


# ---------------------------------------------------------------------------
# Talking to the printer
# ---------------------------------------------------------------------------
def get_config(secrets_obj=None) -> dict:
    """Printer URL and credentials: env vars first, then st.secrets, else empty."""
    def pick(name):
        if os.getenv(name):
            return os.getenv(name)
        try:
            return secrets_obj[name] if secrets_obj is not None and name in secrets_obj else ""
        except Exception:                      # no secrets.toml at all
            return ""
    return {"url": (pick("PRINTER_URL") or "http://172.27.189.79").rstrip("/"),
            "user": pick("PRINTER_USER"), "password": pick("PRINTER_PASS")}


def digest_get(cfg: dict, uri: str) -> Any:
    """GET a printer endpoint. Tries without auth first, then SHA-256 Digest (from data/api.py)."""
    r = requests.get(cfg["url"] + uri, timeout=TIMEOUT_S)
    if r.ok:
        return r.json()
    if r.status_code != 401 or not cfg["user"]:
        r.raise_for_status()

    challenge = r.headers["WWW-Authenticate"]
    realm = re.search(r'realm="([^"]+)"', challenge).group(1)
    nonce = re.search(r'nonce="([^"]+)"', challenge).group(1)
    qop = re.search(r'qop="([^"]+)"', challenge).group(1)
    nc, cnonce = "00000001", secrets.token_hex(16)
    ha1 = hashlib.sha256(f"{cfg['user']}:{realm}:{cfg['password']}".encode()).hexdigest()
    ha2 = hashlib.sha256(f"GET:{uri}".encode()).hexdigest()
    resp = hashlib.sha256(f"{ha1}:{nonce}:{nc}:{cnonce}:{qop}:{ha2}".encode()).hexdigest()
    auth = (f'Digest username="{cfg["user"]}", realm="{realm}", nonce="{nonce}", uri="{uri}", '
            f'algorithm=SHA-256, response="{resp}", qop={qop}, nc={nc}, cnonce="{cnonce}"')
    r = requests.get(cfg["url"] + uri, headers={"Authorization": auth}, timeout=TIMEOUT_S)
    r.raise_for_status()
    return r.json()


def poll_once(cfg: dict) -> dict:
    """One snapshot from the printer as a flat row. Raises on network/HTTP errors."""
    printer = digest_get(cfg, "/api/v1/printer")
    try:
        job = digest_get(cfg, "/api/v1/print_job")
    except requests.HTTPError:
        job = {}                               # the printer answers 404 here when no job is loaded
    return flatten(printer, job)


# ---------------------------------------------------------------------------
# Flattening one snapshot
# ---------------------------------------------------------------------------
def _num(v) -> float:
    if v is None or isinstance(v, bool):
        return np.nan
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


def flatten(printer: dict, job: dict | None) -> dict:
    """/api/v1/printer (+ /api/v1/print_job) JSON -> one flat row. Extruder 1 = heads[0].extruders[0]."""
    job = job or {}
    head = (printer.get("heads") or [{}])[0] or {}
    ex1 = (head.get("extruders") or [{}])[0] or {}
    hotend = ex1.get("hotend") or {}
    stats = hotend.get("statistics") or {}
    ht = hotend.get("temperature") or {}
    bt = (printer.get("bed") or {}).get("temperature") or {}
    pos = head.get("position") or {}
    return {
        "printer_status": printer.get("status", ""),
        "extruder1_temp_current": _num(ht.get("current")),
        "extruder1_temp_target": _num(ht.get("target")),
        "bed_temp_current": _num(bt.get("current")),
        "bed_temp_target": _num(bt.get("target")),
        "head_x_mm": _num(pos.get("x")),
        "head_y_mm": _num(pos.get("y")),
        "head_z_mm": _num(pos.get("z")),
        "head_acceleration": _num(head.get("acceleration")),
        "fan_pct": _num(head.get("fan")),
        "extruder1_material_extruded": _num(stats.get("material_extruded")),
        "extruder1_prints_since_cleaned": _num(stats.get("prints_since_cleaned")),
        "job_name": job.get("name", ""),
        "job_state": job.get("state", ""),
        "job_progress": _num(job.get("progress")),
        "job_time_elapsed": _num(job.get("time_elapsed")),
        "job_time_total": _num(job.get("time_total")),
    }


# ---------------------------------------------------------------------------
# Reading Bella's collector logs (data/api.py output)
# ---------------------------------------------------------------------------
_SECTION = re.compile(r"(/api/v1/[^\s]+)\s*\n=+\s*\n(.*?)(?=\n=+\s*\n/api/v1/|\Z)", re.S)


def parse_log_text(text: str) -> list[dict]:
    """All complete /printer snapshots in a log, each paired with the latest /print_job before it."""
    rows, job = [], {}
    dec = json.JSONDecoder()
    for m in _SECTION.finditer(text):
        endpoint, body = m.group(1), m.group(2)
        start = body.find("{")
        if start < 0:
            continue                           # HTTP error line or half-written section
        try:
            data, _ = dec.raw_decode(body[start:])
        except json.JSONDecodeError:
            continue
        if endpoint == "/api/v1/print_job":
            job = data
        elif endpoint == "/api/v1/printer":
            rows.append(flatten(data, job))
    return rows


# ---------------------------------------------------------------------------
# Rows -> dashboard DataFrame
# ---------------------------------------------------------------------------
def to_frame(rows: list[dict], sample_period_s: float | None = None) -> pd.DataFrame:
    """Add time_s, temperature errors and head speed; mask implausible values.

    time_s is seconds since the first sample: real wall-clock time when polling live,
    sample number x poll period when replaying a log.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col, (lo, hi) in LIMITS.items():
        bad = (df[col] < lo) | (df[col] > hi)
        df.loc[bad, col] = np.nan

    if "t_wall" in df:                         # live polling: real wall-clock seconds
        df["time_s"] = df["t_wall"] - df["t_wall"].iloc[0]
    else:                                      # log replay: the log's timestamps are minutes only,
        period = sample_period_s or _estimate_period(df)   # so space samples by the measured poll period
        df["time_s"] = np.arange(len(df)) * period

    df["nozzle_error_c"] = df["extruder1_temp_current"] - df["extruder1_temp_target"]
    df["bed_error_c"] = df["bed_temp_current"] - df["bed_temp_target"]
    dt = df["time_s"].diff().where(lambda s: s > 0)
    step = np.sqrt(df["head_x_mm"].diff() ** 2 + df["head_y_mm"].diff() ** 2)
    df["head_speed_mm_s"] = (step / dt).where(lambda s: s < 1000)   # average XY speed between samples
    df["printing"] = df["printer_status"].eq("printing") & df["job_state"].eq("printing")
    return df


def _estimate_period(df: pd.DataFrame) -> float:
    """Seconds between samples, from the job clock (about 1.6 s with data/api.py)."""
    e = df["job_time_elapsed"].where(df["job_state"].eq("printing")).dropna()
    if len(e) > 10 and e.iloc[-1] > e.iloc[0]:
        return float((e.iloc[-1] - e.iloc[0]) / (len(e) - 1))
    return 1.6
