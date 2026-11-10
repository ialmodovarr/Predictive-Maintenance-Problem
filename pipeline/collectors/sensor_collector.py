import math
import time
from uuid import uuid4
from legacy.data.sensors.print_logger import find_port
from pipeline.synchronization.timestamp_sync import DeviceClock


def to_m_s2(counts, config):
    """Raw X/Y/Z counts -> m/s^2, with gravity removed from X when configured.

    Raw counts are stored alongside, so the uncorrected values stay recoverable.
    """
    scale = config.gravity_m_s2/config.counts_per_g
    x, y, z = (v*scale for v in counts)
    return [x - config.x_gravity_offset_m_s2, y, z]


def parse(line, clock, received, config):
    parts = line.split(",")
    if len(parts) not in (6, 7):
        raise ValueError("expected 6 or 7 sensor fields")
    micros = len(parts) == 7
    tick = int(parts[0])
    seq = int(parts[1]) if micros else None
    x, y, z, count = map(int, parts[-5:-1])
    temperature = float(parts[-1])
    if tick < 0 or (micros and (tick >= 2**32 or seq < 0)) or any(not -32768<=v<=32767 for v in (x,y,z)):
        raise ValueError("invalid sensor count or clock")
    previous_sequence = clock.seq
    t, segment, sync = clock.map(tick if micros else tick*1000, seq, received, micros)
    return t, {"line": line, "device_tick": tick, "tick_unit": "us" if micros else "ms",
               "invalidates_segment": clock.invalidates_segment,
               "missing_packets_before": max(0, seq-previous_sequence-1)
                   if seq is not None and previous_sequence is not None else 0,
               "sequence": seq, "segment": segment, "sync": sync,
               "counts": [x,y,z], "filament_counts": count,
               "accel_m_s2": to_m_s2((x,y,z), config),
               "displacement": config.feed_direction*count*config.mm_per_turn/4096,
               "temperature": temperature if math.isfinite(temperature) else None}


def collect(config, emit, stop):
    import serial
    clock = DeviceClock(config.clock_jump_tolerance_s, config.sensor_gap_s)
    connection = uuid4().hex
    with serial.Serial(config.port or find_port(), config.baud, timeout=.1) as port:
        if hasattr(port, "set_buffer_size"):
            port.set_buffer_size(rx_size=1 << 20)
        pending = b""
        while not stop.is_set():
            pending += port.read(min(port.in_waiting or 1, 65536))
            lines = pending.split(b"\n")
            pending = lines.pop()
            if len(pending) > 65536:
                raise ValueError("unterminated serial record")
            received = time.time()
            # USB can deliver many measurements at once. Anchor the newest tick
            # in each contiguous batch to receipt and retain native sample spacing.
            anchors = [received] * len(lines)
            previous = None
            for i in range(len(lines)-1, -1, -1):
                fields = lines[i].split(b",")
                if len(fields) not in (6, 7) or not fields[0].isdigit():
                    previous = None
                    continue
                tick = int(fields[0]) * (1 if len(fields)==7 else 1000)
                if previous is not None and 0 <= previous-tick < 1000000:
                    anchors[i] = anchors[i+1] - (previous-tick)/1e6
                previous = tick
            for raw,anchor in zip(lines,anchors):
                line = raw.decode("ascii", errors="backslashreplace").rstrip("\r")
                try:
                    t, record = parse(line, clock, anchor, config)
                    record["segment"] = f"{connection}:{record['segment']}"
                    if record["invalidates_segment"] is not None:
                        record["invalidates_segment"] = f"{connection}:{record['invalidates_segment']}"
                    if record["sync"] in ("forward_clock_discontinuity", "acquisition_gap", "device_reset"):
                        print(f"STM32 timing: {record['sync']}; segment {record['segment']}", flush=True)
                    record["receipt_minus_mapped_s"] = received-t
                except ValueError as exc:
                    t, record = received, {"line": line, "error": str(exc)}
                    if "zeroed" in line.lower():
                        # The firmware's zero button changes cumulative counts.
                        # Start a fresh segment before deriving displacement rates.
                        clock.segment += 1
                    print("STM32:", line[:160], flush=True)
                emit(t, received, record)
