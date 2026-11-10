import csv
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import time
import unittest
from unittest.mock import MagicMock, patch
import sys

import numpy as np
from pipeline.collectors import api_collector, sensor_collector
from pipeline.config import Config
from pipeline.features import api_features, sensor_features
from pipeline.labels.label_manager import prompt_labels, COMPONENTS
from pipeline.runtime import record
from pipeline.storage.csv_writer import allocate
from pipeline.synchronization.timestamp_sync import DeviceClock
from pipeline.windowing.rolling_window import Scheduler


def printer(t):
    head = {"position":{"x":3*t,"y":4*t,"z":0},
            "extruders":[{"hotend":{"temperature":{"current":200+t,"target":200}}}]}
    return {"endpoint":"/api/v1/printer","payload":{"heads":[head]}}


def read(path):
    with path.open(newline="",encoding="utf8") as f:
        return list(csv.DictReader(f))


class PipelineTests(unittest.TestCase):
    def test_filament_activity_uses_time_and_breaks_at_segment_boundaries(self):
        times = [0, 1, 3, 4, 5, 6, 7, 8, 9, 9.5]
        positions = [0, 1, 0, 0, 20, 20, 20, 20, 20, 20]
        rows = [(t, {"sequence":i, "device_tick":int(1e6*t), "segment":int(t>=5),
                     "sync":"test", "accel":[0,0,0], "temperature":20,
                     "displacement":positions[i]}) for i,t in enumerate(times)]
        config = Config(sensor_gap_s=2)
        out,_ = sensor_features.extract(rows,config)
        self.assertAlmostEqual(out[sensor_features.MOVEMENT_PERCENT],30)
        self.assertAlmostEqual(out[sensor_features.LONGEST_IDLE],4.5)
        sparse,_ = sensor_features.extract(rows[:2],config)
        for feature in (sensor_features.MOVEMENT_PERCENT,sensor_features.LONGEST_IDLE):
            self.assertTrue(math.isnan(sparse[feature]))

    def test_nominal_800_hz_accepts_actual_765_hz(self):
        times = np.arange(7650)/765
        values = np.sin(2*np.pi*30*times)
        out = {}
        sensor_features.acceleration(out,"accel_x_",times,values,Config())
        self.assertAlmostEqual(out["accel_x_dominant_frequency"],30,delta=.1)

    def test_run2_forward_jump_invalidates_pending_segment(self):
        clock = DeviceClock()
        c = Config()
        scheduler = Scheduler(100, c)
        before = "724673748,557656,224,89,-118,38,61.12"
        after = "1075758389,827255,267,32,-20,56,57.93"
        t, old = sensor_collector.parse(before, clock, 100.05, c)
        scheduler.add("sensor",t,100.05,old)
        mapped, current = sensor_collector.parse(after, clock,100.04675,c)
        self.assertEqual(current["sync"],"forward_clock_discontinuity")
        self.assertEqual(current["segment"],1)
        self.assertEqual(current["invalidates_segment"],0)
        self.assertGreaterEqual(mapped,t)
        scheduler.add("sensor",mapped,100.05,current)
        _,_,_,data,diagnostics = scheduler.take(110.5)
        self.assertEqual(data["sensor"],[(mapped,current)])
        self.assertEqual(diagnostics["sensor"]["timing_excluded_total"],1)

    def test_run2_acquisition_gap_keeps_native_elapsed_time(self):
        clock = DeviceClock()
        c = Config()
        first,r1 = sensor_collector.parse(
            "1206759494,927761,230,-22,46,127,61.18",clock,100,c)
        second,r2 = sensor_collector.parse(
            "1207385259,928241,318,3,-51,127,61.25",clock,100.7,c)
        self.assertAlmostEqual(second-first,.625765)
        self.assertEqual(r2["sync"],"acquisition_gap")
        self.assertEqual(r2["missing_packets_before"],479)
        self.assertNotEqual(r1["segment"],r2["segment"])

    def test_unknown_rate_does_not_bridge_gap(self):
        rows=[]
        for i in range(100):
            t=i/10
            if 40<=i<=46:
                continue
            rows.append((t,{"sequence":i,"segment":0,"sync":"receipt_offset_unverified",
                           "device_tick":i*100000,"accel":[1,1,1],
                           "temperature":20+t,"displacement":0 if i<40 else 100}))
        out,diag = sensor_features.extract(rows,Config(accel_hz=0))
        self.assertEqual(out["filament_velocity_mean"],0)
        self.assertLess(diag["filament_observed_s"],9.2)
        self.assertEqual(out[sensor_features.MOVEMENT_PERCENT],0)
        self.assertAlmostEqual(out[sensor_features.LONGEST_IDLE],5.2)
        self.assertTrue(math.isnan(out["thermocouple_temperature_slope"]))
        self.assertTrue(math.isfinite(out["thermocouple_temperature_mean"]))

    def test_labels(self):
        answers = iter(["healthy","n","failing","","2","3,1","yes"])
        labels = prompt_labels(lambda _:next(answers))
        self.assertEqual(list(labels.values()),["failing","healthy","failing"])
        answers = iter(["1","y"])
        self.assertEqual(set(prompt_labels(lambda _:next(answers)).values()),{"healthy"})

    def test_run_allocation(self):
        with TemporaryDirectory() as d:
            first = allocate(Path(d))
            marker = first/"raw.txt"
            marker.write_text("original")
            self.assertEqual(allocate(Path(d)).name,"run_0002")
            self.assertEqual(marker.read_text(),"original")

    def test_device_protocol(self):
        c = Config()
        clock = DeviceClock()
        t,r = sensor_collector.parse("100000,1200,256,0,-256,4096,nan",clock,1700000000,c)
        self.assertEqual(t,1700000000)
        self.assertEqual(r["accel"],[1,0,-1])
        self.assertIsNone(r["temperature"])
        self.assertEqual(r["displacement"],c.mm_per_turn)
        t2,_,_ = clock.map(101250,1201,t+.01,True)
        self.assertAlmostEqual(t2-t,.00125,places=6)
        self.assertEqual(clock.map(100500,1200,t+.02,True)[2],"out_of_order")
        self.assertEqual(clock.map(0,0,t+10,True)[1:],(1,"device_reset"))
        old = DeviceClock()
        _,r = sensor_collector.parse("120,0,0,0,12,20",old,10,c)
        self.assertEqual(r["tick_unit"],"ms")
        wrap = DeviceClock()
        wrap.map(2**32-100,3000,0,True)
        self.assertAlmostEqual(wrap.map(100,3001,1,True)[0],.0002)
        with self.assertRaises(ValueError):
            sensor_collector.parse("1,3,99999,2,3,4,5",clock,t,c)

    def test_window_boundaries(self):
        s = Scheduler(100,Config())
        for t,receipt in [(110,110),(103,103),(100,100),(104,111)]:
            s.add("api",t,receipt,printer(t))
        self.assertIsNone(s.take(110.49))
        i,start,end,data,diag = s.take(110.5)
        self.assertEqual((i,start,end),(1,100,110))
        self.assertEqual(sorted(t for t,_ in data["api"]),[100,103])
        self.assertEqual(diag["api"]["late_excluded"],1)
        self.assertIsNone(s.take(110.5))
        i,start,end,data,_ = s.take(112,True)
        self.assertEqual((i,start,end),(2,102,112))
        self.assertEqual(sorted(t for t,_ in data["api"]),[103,104,110])
        self.assertEqual(data["sensor"],[])

    def test_api_features(self):
        times = [0,1,2,4,6,8,9]
        rows = [(t,printer(t)) for t in times]
        out,_ = api_features.extract(rows,Config())
        expected = [200+np.mean(times),np.std(times,ddof=1),1,np.mean(times),5,0]
        np.testing.assert_allclose(list(out.values()),expected)
        rows.append((1,printer(1)))
        self.assertEqual(api_features.extract(rows,Config())[0],out)
        rows[0][1]["payload"]["heads"][0]["position"]["x"] = None
        self.assertLess(api_features.extract(rows,Config())[1]["head_observed_s"],9)
        out,diag = api_features.extract([],Config())
        self.assertEqual(set(out),set(diag["missing"]))

    def test_sensor_features(self):
        c = Config(accel_hz=100)
        rows = []
        for i in range(1000):
            t=i/100
            a=np.sin(2*np.pi*7*t)
            rows.append((t,{"accel":[a,2*a,3*a],"sequence":i,"device_tick":10000*i,
                            "sync":"test","segment":0,"temperature":20+2*t,
                            "displacement":-2*t}))
        out,diag = sensor_features.extract(rows,c)
        self.assertEqual(len(out),22)
        for k,axis in enumerate("xyz",1):
            n=len(rows)
            expected=[k/math.sqrt(2),k*math.sqrt(n/(2*(n-1))),2*k,7,
                      (n-1)*((n+1)*(-1.5)+6)/((n-2)*(n-3))]
            actual=[out[f"accel_{axis}_{m}"] for m in sensor_features.METRICS]
            np.testing.assert_allclose(actual,expected,atol=.01)
        np.testing.assert_allclose([out[k] for k in sensor_features.FEED],[-2,0],atol=1e-8)
        temperatures=[r["temperature"] for _,r in rows]
        np.testing.assert_allclose([out[sensor_features.TEMP+"_"+k] for k in ("mean","std","slope")],
                                   [np.mean(temperatures),np.std(temperatures,ddof=1),2])
        self.assertFalse(diag["missing"])
        out,diag = sensor_features.extract(rows,Config(accel_hz=0))
        self.assertTrue(math.isfinite(out["accel_x_rms"]))

    def test_hardware_adapters(self):
        stop=Event()
        session=MagicMock()
        response=session.get.return_value
        response.json.return_value={"sample":1}
        emitted=[]
        def accept(t,received,row):
            emitted.append(row)
            if len(emitted)==2:
                stop.set()
        with patch("requests.Session") as factory:
            factory.return_value.__enter__.return_value=session
            api_collector.collect(Config(printer_url="http://local"),accept,stop)
        self.assertEqual([r["endpoint"] for r in emitted],["/api/v1/print_job","/api/v1/printer"])
        stop.clear()
        emitted.clear()
        serial=MagicMock()
        port=serial.Serial.return_value.__enter__.return_value
        port.in_waiting=100
        port.read.side_effect=[b"10,1,0,0,",b"256,0,20\n20,2,0,0,256,0,20\n"]
        with patch.dict(sys.modules,{"serial":serial}):
            sensor_collector.collect(Config(port="mock"),accept,stop)
        self.assertEqual([r["sequence"] for r in emitted],[1,2])

    def test_segment_boundary_preserves_axis_statistics(self):
        rows = [(i, {"sequence":i,"segment":i//5,"sync":"device_reset",
                    "device_tick":i%5,"accel":[1,1,1],"temperature":22,
                    "displacement":i%5}) for i in range(10)]
        out,diag = sensor_features.extract(rows,Config(accel_hz=0,accel_count=4,sensor_gap_s=2))
        self.assertEqual(out["accel_x_rms"],1)
        self.assertEqual(out["accel_x_std"],0)
        self.assertTrue(math.isnan(out["accel_x_kurtosis"]))
        self.assertEqual(out["filament_velocity_mean"],1)
        self.assertEqual(diag["filament_observed_s"],8)
        self.assertTrue(math.isnan(out["thermocouple_temperature_slope"]))
        self.assertEqual(out["thermocouple_temperature_mean"],22)

    def test_conflicting_snapshot_and_missing_target(self):
        rows = [(i,printer(i)) for i in range(10)]
        for _,r in rows:
            del r["payload"]["heads"][0]["extruders"][0]["hotend"]["temperature"]["target"]
        rows.append((1,printer(25)))
        out,diag = api_features.extract(rows,Config())
        self.assertTrue(math.isnan(out["api_extruder_temperature_error"]))
        self.assertEqual(diag["temperature"]["count"],9)
        self.assertAlmostEqual(out["api_extruder_temperature_slope"],1)

    def test_overflow_reported_and_expired_data(self):
        s = Scheduler(0,Config(buffer_limit=2))
        for t in (1,2,3):
            s.add("sensor",t,t,{})
        _,_,_,data,diag = s.take(10.5)
        self.assertEqual(len(data["sensor"]),2)
        self.assertEqual(diag["sensor"]["buffer_evictions_total"],1)
        s.add("sensor",1,11,{})
        self.assertEqual(s.take(12.5)[-1]["sensor"]["arrived_after_eviction_total"],1)

    def test_delayed_and_partial_stop(self):
        s = Scheduler(0,Config())
        self.assertIsNone(s.take(9.99,True))
        self.assertTrue(s.take(10.1,True)[-1]["api"]["shortened_wait"])
        self.assertIsNone(s.take(10.1,True))

    def test_live_files_and_disconnect(self):
        with TemporaryDirectory() as d:
            root=Path(d)
            stop=Event()
            retries=[]
            original={"line":"raw board status","error":"diagnostic"}
            def sensor(c,emit,halt):
                retries.append(1)
                if len(retries)==1:
                    raise OSError("simulated disconnect")
                t=time.time()
                emit(t,t,original)
                halt.wait()
            def api(c,emit,halt):
                while not halt.is_set():
                    t=time.time()
                    emit(t,t,printer(1))
                    halt.wait(.03)
            failures=[]
            def run():
                try:
                    record(root,dict.fromkeys(COMPONENTS,"healthy"),Config(window=.2,stride=.1,allowance=.02),
                           stop,{"api":api,"sensor":sensor})
                except Exception as exc:
                    failures.append(exc)
            th=Thread(target=run)
            th.start()
            directory=root/"run_0001"
            path=directory/"features"/"sensor_features.csv"
            try:
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    if len(retries)>=2 and path.exists() and len(read(path))>=3:
                        break
                    time.sleep(.05)
                live=read(path)
                self.assertGreaterEqual(len(live),3)
            finally:
                stop.set()
                th.join(5)
            self.assertFalse(th.is_alive())
            self.assertFalse(failures)
            self.assertEqual(len(list(directory.rglob("*.csv"))),5)
            self.assertEqual(json.loads(read(directory/"raw"/"sensor_raw.csv")[0]["record_json"]),original)
            a=read(directory/"features"/"api_features.csv")
            b=read(path)
            metadata = set(COMPONENTS) | {"run_id","window_id","window_start_utc","window_end_utc",
                                      "processing_delay_s","diagnostics"}
            self.assertEqual(set(b[0])-metadata,set(sensor_features.NAMES))
            self.assertEqual(len(set(b[0])-metadata),22)
            self.assertEqual(json.loads((directory/"settings.json").read_text())["feature_version"],3)
            self.assertEqual(b[:len(live)],live)
            self.assertEqual(len(a),len(b))
            self.assertEqual(len({r["window_id"] for r in a}),len(a))
            labels=read(directory/"labels"/"print_labels.csv")[0]
            for left,right in zip(a,b):
                for key in (*labels,"window_id","window_start_utc","window_end_utc"):
                    self.assertEqual(left[key],right[key])
                for key,value in labels.items():
                    self.assertEqual(left[key],value)
                self.assertNotIn("window_quality",left)
                self.assertGreaterEqual(float(left["processing_delay_s"]),0)
