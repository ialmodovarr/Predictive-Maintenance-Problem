"""Segmented device clock with a fixed receipt-based offset per clock epoch."""
from datetime import datetime, timezone


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class DeviceClock:
    def __init__(self, jump_tolerance=0.5, gap_limit=0.25):
        self.jump_tolerance = jump_tolerance
        self.gap_limit = gap_limit
        self.invalidates_segment = None
        self.last_device = None
        self.last_mapped = None
        self.previous = None
        self.seq = None
        self.wrap = 0
        self.segment = 0
        self.offset = None

    def map(self, tick, seq, received, micros):
        self.invalidates_segment = None
        previous_seq = self.seq
        status = "receipt_offset_unverified"
        if self.previous is not None:
            backwards = self.previous - tick
            restart = backwards > 0 and (seq is None and backwards > 1000000 or
                                          seq is not None and seq < self.seq and
                                          (seq == 0 or (seq < 1000 and backwards > 1000000)))
            if restart:
                self.segment += 1
                self.offset = None
                self.wrap = 0
                status = "device_reset"
            elif micros and backwards > 2**31:
                self.wrap += 2**32
            elif backwards > 0 or (seq is not None and seq <= self.seq):
                return (tick+self.wrap)/1e6+self.offset, self.segment, "out_of_order"
        device = (tick+self.wrap)/1e6
        if self.offset is not None:
            if device+self.offset > received+self.jump_tolerance:
                self.invalidates_segment = self.segment
                self.segment += 1
                self.offset = None
                status = "forward_clock_discontinuity"
            elif self.last_device is not None and (
                device-self.last_device > self.gap_limit or
                seq is not None and previous_seq is not None and seq > previous_seq+1
            ):
                self.segment += 1
                status = "acquisition_gap"
        self.previous, self.seq = tick, seq
        self.last_device = device
        # Keep the mapping fixed within a clock epoch so buffering cannot
        # compress acquisition gaps or move later measurements backwards.
        if self.offset is None:
            if self.last_mapped is not None:
                received = max(received, self.last_mapped)
            self.offset = received-device
        self.last_mapped = device+self.offset
        return device+self.offset, self.segment, status
