"""Connection and provisional feature settings, overridable via JSON."""
from dataclasses import dataclass, fields
import json
import math
from pathlib import Path


PRINTER_USERNAME = "6bf926c99944ad9e15da49c2ed1bffb5"
PRINTER_PASSWORD = "716b548893f2fe16f0d1929b94857484d291b8af67e61685ba9c1ad24f682e30"


@dataclass(frozen=True)
class Config:
    printer_url: str = "http://172.27.189.79"
    port: str = ""
    baud: int = 921600
    counts_per_g: float = 256
    # Acceleration is stored and featurized in m/s^2: counts/counts_per_g*gravity_m_s2.
    # x_gravity_offset_m_s2 is then subtracted from X only, the axis aligned with
    # gravity on the installed mount. Set it to 0 to keep X uncorrected.
    gravity_m_s2: float = 9.81
    x_gravity_offset_m_s2: float = 9.81
    mm_per_turn: float = 50.270
    feed_direction: int = 1
    window: float = 10
    stride: float = 2
    allowance: float = 0.5
    poll: float = 1
    timeout: float = 2
    api_count: int = 5
    api_span: float = 7.5
    head_intervals: int = 4
    head_span: float = 7
    head_gap: float = 2.5
    accel_count: int = 32
    accel_span: float = 9
    fft_count: int = 256
    accel_hz: float = 800
    accel_coverage: float = .85
    fft_coverage: float = .95
    fft_min_span: float = 5
    fft_gap_s: float = .00375
    gap_periods: float = 3
    filament_hz: float = 0
    filament_intervals: int = 4
    filament_span: float = 8
    temp_hz: float = 0
    temp_count: int = 5
    temp_span: float = 8
    temp_slope_gap_s: float = 1
    buffer_limit: int = 100000
    clock_jump_tolerance_s: float = 0.5
    sensor_gap_s: float = 0.25

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(f.default, str):
                if not isinstance(v, str):
                    raise ValueError(f.name)
            elif f.name == "feed_direction":
                if v not in (-1, 1):
                    raise ValueError(f.name)
            elif not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError(f.name)
            elif v == 0 and f.name not in {"allowance", "accel_hz", "temp_hz", "filament_hz", "x_gravity_offset_m_s2"}:
                raise ValueError(f.name)
            if f.type is int and type(v) is not int:
                raise ValueError(f.name)
        if max(self.accel_coverage, self.fft_coverage) > 1:
            raise ValueError("Coverage must be a fraction")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text(encoding="utf-8-sig"))) if path else cls()
