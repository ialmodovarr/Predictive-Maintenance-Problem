"""Synthetic tests for the historical replay's timing reconstruction."""
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.replay import historical as h

TZ = ZoneInfo("America/New_York")


class ApiTimeTests(unittest.TestCase):
    def test_minute_stamps_spread_within_minute(self):
        start = datetime(2026, 10, 30, 14, 40, 55, tzinfo=TZ)
        stamps = ["14:40"]*3 + ["14:41"]*30 + ["14:42"]*30 + ["14:43"]*10
        times = h.estimate_api_times(stamps, start)
        self.assertTrue(all(b > a for a, b in zip(times, times[1:])))
        self.assertGreaterEqual(times[0], start.timestamp())
        minute = datetime(2026, 10, 30, 14, 41, tzinfo=TZ).timestamp()
        self.assertTrue(all(minute <= t < minute + 60 for t in times[3:33]))
        # Last minute uses the interior period (2 s), so 10 stamps span ~20 s.
        self.assertAlmostEqual(times[-1] - times[-10], 18.0, places=6)

    def test_second_resolution_stamps_used_directly(self):
        start = datetime(2026, 11, 10, 9, 0, 0, tzinfo=TZ)
        times = h.estimate_api_times(["09:00:01.500", "09:00:02.750"], start)
        self.assertAlmostEqual(times[1] - times[0], 1.25)
        self.assertAlmostEqual(times[0] - start.timestamp(), 1.5)

    def test_file_names_converted_from_local_time(self):
        self.assertEqual(h.file_start("api_data_2026-10-30_14-40-55.txt", TZ).utcoffset().total_seconds(), -4*3600)
        self.assertEqual(h.file_start("print_20261104_113531.csv", TZ).utcoffset().total_seconds(), -5*3600)


class SensorRepairTests(unittest.TestCase):
    def write_log(self, directory, ticks, seqs):
        n = len(ticks)
        frame = pd.DataFrame({
            "time_s": np.arange(n)*0.00125, "accel_x_g": 1.0, "accel_y_g": 0.0, "accel_z_g": 0.0,
            "filament_mm": 0.0, "hotend_temp_C": np.nan, "board_t_us": ticks, "sample_index": seqs,
            "accel_x_counts": 256, "accel_y_counts": 0, "accel_z_counts": 0, "filament_counts": 0})
        path = Path(directory)/"print_20261030_144152.csv"
        frame.to_csv(path, index=False)
        return path

    def records(self, path):
        stats = {}
        recs = list(h.sensor_records(path, TZ, Config(), stats))
        return recs, stats["sensor_files"][path.name]

    def test_spurious_wrap_offsets_and_corrupted_rows(self):
        n = 2000
        ticks = (10_000_000 + np.arange(n)*1250).astype(np.int64)
        seqs = np.arange(n, dtype=np.int64) + 500
        ticks[1000:] += 2**32                     # logger's false wrap
        ticks[700] = 64_865_019_003                # garbled clock
        seqs[1500] += 99_999                       # garbled sample number
        with tempfile.TemporaryDirectory() as d:
            recs, stats = self.records(self.write_log(d, ticks, seqs))
        rejected = [r for r in recs if "error" in r[3]]
        valid = [r for r in recs if "error" not in r[3]]
        self.assertEqual(sorted(int(r[3]["replay_source"].split(":")[1]) for r in rejected), [700, 1500])
        self.assertEqual(stats["clock_repaired_rows"], 1001)   # 1000 false-wrap rows + the garbled one
        times = np.array([t for t, *_ in valid])
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertAlmostEqual(times[-1] - times[0], (n-1)*0.00125, places=6)
        self.assertNotIn("out_of_order", {r[3]["sync"] for r in valid})
        self.assertEqual(valid[0][3]["accel_m_s2"], [0.0, 0.0, 0.0])

    def test_genuine_gap_is_kept(self):
        ticks = np.r_[np.arange(500)*1250, 114_000_000 + np.arange(500)*1250].astype(np.int64)
        seqs = np.r_[np.arange(500), 91_200 + np.arange(500)].astype(np.int64)
        with tempfile.TemporaryDirectory() as d:
            recs, stats = self.records(self.write_log(d, ticks, seqs))
        self.assertEqual(stats["rejected_rows"], 0)
        self.assertIn("acquisition_gap", {r[3]["sync"] for r in recs})


if __name__ == "__main__":
    unittest.main()
