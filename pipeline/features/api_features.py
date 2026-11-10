import math
import numpy as np
from .common import number, series, temperatures, support, intervals, velocity, missing

TEMP = "api_extruder_temperature"
HEAD = ("api_head_velocity_mean", "api_head_velocity_std")
NAMES = tuple(TEMP+"_"+s for s in ("mean","std","slope","error")) + HEAD


def extract(records, config):
    out = dict.fromkeys(NAMES, math.nan)
    rows = []
    for t, record in records:
        if record["endpoint"] != "/api/v1/printer":
            continue
        payload = record["payload"]
        try:
            head = payload["heads"][0]
        except (TypeError, KeyError, IndexError):
            head = {}
        try:
            temp = head["extruders"][0]["hotend"]["temperature"]
        except (KeyError, IndexError, TypeError):
            temp = {}
        current = number(temp.get("current")) if isinstance(temp, dict) else math.nan
        target = number(temp.get("target")) if isinstance(temp, dict) else math.nan
        pos = head.get("position") if isinstance(head, dict) else None
        rows.append((t, {**(pos if isinstance(pos,dict) else {}), "temp":current,"error":current-target}))
    t,x = series(rows,"temp")
    diagnostics = {"temperature": temperatures(out,TEMP,t,x,config.api_count,config.api_span)}
    _, errors = series(rows,"error")
    valid = np.isfinite(errors)
    diagnostics["paired_error"] = support(t[valid])
    if valid.sum()>=config.api_count and diagnostics["paired_error"]["span_s"]>=config.api_span:
        out[TEMP+"_error"] = float(errors[valid].mean())
    positions = np.column_stack([series(rows,k)[1] for k in "xyz"])
    dt,v,_ = intervals(t,positions,config.head_gap)
    velocity(out,HEAD,dt,v,config.head_intervals,config.head_span)
    diagnostics.update(head_intervals=len(dt),head_observed_s=float(dt.sum()),missing=missing(out,{}))
    return out,diagnostics
