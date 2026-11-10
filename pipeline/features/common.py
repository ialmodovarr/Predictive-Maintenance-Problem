"""Numeric operations on time-ordered observations inside a single window."""
import math
import numpy as np


def number(value):
    if isinstance(value, bool):
        return math.nan
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        return math.nan
    return value if math.isfinite(value) else math.nan


def series(rows, key):
    """Collapse identical timestamp/value pairs, mask conflicting observations."""
    groups = {}
    for t, row in rows:
        groups.setdefault(t, set()).add(number(row.get(key)))
    t = np.array(sorted(groups), dtype=float)
    x = np.array([next(iter(groups[v])) if len(groups[v]) == 1 else math.nan for v in t])
    return t, x


def support(t):
    return {"count": len(t), "span_s": float(np.ptp(t)) if len(t) else 0.,
            "max_gap_s": float(np.max(np.diff(t))) if len(t)>1 else None}


def temperatures(out, prefix, t, x, count, span, gap=math.inf):
    valid = np.isfinite(x)
    t, x = t[valid], x[valid]
    stats = support(t)
    if len(x)>=count and stats["span_s"]>=span and (stats["max_gap_s"] or 0)<=gap:
        out[prefix+"_mean"] = float(x.mean())
        if len(x)>1:
            out[prefix+"_std"] = float(x.std(ddof=1))
        if len(x)>=3:
            dt = t-t[0]
            dt -= dt.mean()
            out[prefix+"_slope"] = float(np.dot(dt, x-x.mean())/np.dot(dt,dt))
    return stats


def intervals(t, positions, gap, signed=False):
    dt = np.diff(t)
    changes = np.diff(positions, axis=0)
    distance = changes if signed else np.linalg.norm(changes, axis=1)
    valid = np.isfinite(distance) & (dt>0) & (dt<=gap)
    return dt[valid], distance[valid]/dt[valid], valid


def velocity(out, names, dt, v, count, span):
    if len(dt)>=count and sum(dt)>=span:
        mean = np.average(v, weights=dt)
        out[names[0]] = float(mean)
        out[names[1]] = float(np.sqrt(np.average((v-mean)**2, weights=dt)))


def missing(out, reasons):
    return {k: reasons.get(k, "too_few_valid_samples_or_coverage") for k,v in out.items() if not math.isfinite(v)}
