"""Generate raw mock recordings and replay through the production window pipeline."""
import argparse
import csv
import heapq
import json
import math
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from pipeline.config import Config
from pipeline.runtime import write_window
from pipeline.storage.csv_writer import Store
from pipeline.synchronization.timestamp_sync import utc
from pipeline.windowing.rolling_window import Scheduler

DATA = Path(__file__).resolve().parents[1] / "data"
REAL = DATA / "runs"
MOCK = DATA / "mock_runs"
META = DATA / "mock_metadata"
TEST = {3, 7, 9, 12, 19, 27}


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def original(path, source):
    if path.exists():
        with (path / "raw" / (source + "_raw.csv")).open(newline="") as f:
            for row in csv.DictReader(f):
                yield timestamp(row["timestamp_utc"]), timestamp(row["receipt_utc"]), source, json.loads(row["record_json"]), row


def head(record):
    return record["payload"]["heads"][0]


def motion(n, length, rng, api, start, severity):
    """One-second operating trajectory shared by all synthetic streams."""
    t = np.arange(math.ceil(length) + 2)
    phase = np.cumsum(np.repeat(rng.choice([0, .4, .9, 1.5], len(t)//6+1, p=[.08,.38,.38,.16]),6)[:len(t)])
    x = 110 + rng.uniform(8,22)*np.sin(phase)
    y = 90 + rng.uniform(7,20)*np.sin(.67*phase + rng.random())
    target = rng.choice([195.,200.,205.])
    thermal = np.zeros(len(t))
    for i in range(1,len(t)):
        thermal[i] = .95*thermal[i-1] + rng.normal(0,.19)
    temp = target + thermal + severity*np.sin(t/23)*(.5+.5*np.sin(t/83))
    z = 1 + .2*np.floor(t/25)
    printers = [(s-start,head(r)) for s,_,_,r,_ in api if r["endpoint"].endswith("/printer") and r["payload"].get("heads")]
    if printers:
        ts, heads = zip(*printers)
        for values, key in [(x,"x"),(y,"y"),(z,"z"),(temp,None)]:
            obs = [h["position"][key] if key else h["extruders"][0]["hotend"]["temperature"]["current"] for h in heads]
            last = min(len(t)-1,int(ts[-1]))
            values[last+1:] += obs[-1]-values[last]
            values[:last+1] = np.interp(t[:last+1],ts,obs)
        target = heads[-1]["extruders"][0]["hotend"]["temperature"]["target"]
    speed = np.hypot(np.gradient(x),np.gradient(y))
    rate = np.where(speed<.6,0,np.minimum(.55,speed*.032)) * (1-.25*severity/(1+severity)*np.sin(t/13)**4)
    rate[(np.arange(len(t)) % 47)==0] = -.32
    return t,x,y,z,temp,target,speed,np.cumsum(rate)


def api_stream(api, state, start, length, n, rng):
    yield from api
    templates = {}
    for row in api or original(REAL/"run_0001","api"):
        r = row[3]
        if isinstance(r["payload"],dict) and (r["payload"].get("heads") or r["payload"].get("state")=="printing"):
            templates[r["endpoint"]] = r["payload"]
        if not api and len(templates)==2: break
    t,x,y,z,temp,target,_,feed = state
    begin = int(api[-1][0]-start)+1 if api else 0
    job = templates["/api/v1/print_job"]
    elapsed = job.get("time_elapsed",0) if api else 0
    initial_material = templates["/api/v1/printer"]["heads"][0]["extruders"][0]["hotend"]["statistics"]["material_extruded"]
    for i in range(begin, math.ceil(length)):
        for k,endpoint in enumerate(("/api/v1/print_job","/api/v1/printer")):
            s = float(start+i+.03+rng.uniform(0,.06)+.22*k)
            if s>start+length: continue
            p = templates[endpoint]
            if k:
                h = p["heads"][0]
                h["position"].update(x=round(x[i],3),y=round(y[i],3),z=round(z[i],3))
                hot = h["extruders"][0]["hotend"]
                hot["temperature"].update(current=round(temp[i],1),target=float(target))
                hot["statistics"]["material_extruded"] = initial_material+int(max(0,feed[i]-feed[begin]))
            else:
                total = max(job.get("time_total",0) if api else 0, elapsed+length-begin+30)
                e = int(elapsed+i-begin)
                p.update(time_elapsed=e,time_total=int(total),time_estimated=int(total),progress=round(e/total,5),state="printing")
                if not api: p.update(datetime_started=utc(start),uuid=f"00000000-0000-4000-8000-{n:012d}")
            yield s,s,"api",{"endpoint":endpoint,"payload":json.loads(json.dumps(p))},None


def synth_sensor(lo,hi,start,state,rng,params,n,base=None):
    """Generate packets in one-second blocks with quantized and held measurements."""
    t,x,y,z,temp,target,speed,feed = state
    hz = params["packet_hz"]
    freq = rng.choice([61,135,191])+rng.normal(0,2)
    offset = np.array(base["counts"] if base else [259,24,-4])
    offset = np.clip(offset,[225,-50,-50],[285,60,60])
    thermal = (base or {}).get("temperature") or rng.normal(67.5,1.4)
    count = (base or {}).get("filament_counts",0)
    seq = (base or {}).get("sequence",0) or 0
    tick = (base or {}).get("device_tick",0)
    baseline = np.interp(lo,t,feed)
    phase = rng.uniform(0,2*np.pi)
    envelope_phase = rng.uniform(0,2*np.pi)
    resonance_period = rng.uniform(8,19)
    for first in range(0,math.ceil((hi-lo)*hz),hz):
        indices = np.arange(first,min(first+hz,math.ceil((hi-lo)*hz)))
        u = lo+(indices+rng.uniform(.03,.12,len(indices)))/hz
        sp = np.interp(u,t,speed)
        amp = (32+np.minimum(sp,35)*.35+11*np.sin(u/resonance_period+envelope_phase)+7*np.sin(u/4.8))[:,None]
        mode = freq if np.median(sp)>7 else 135.0
        angle = phase+2*np.pi*mode*(u-u[0])
        phase = (phase+2*np.pi*mode*len(u)/hz)%(2*np.pi)
        waves = np.sin(angle[:,None]+[0,.8,1.9]) + .4*np.sin(2*np.pi*u[:,None]*(freq*.72)+[.3,1,2])
        counts = np.rint(offset+amp*waves+rng.normal(0,7,(len(u),3))).astype(int).tolist()
        temps = thermal + .18*(np.interp(np.floor(u*3)/3,t,temp)-np.interp(lo,t,temp))+.45*np.sin((u-lo)/38)
        feeds = np.rint(count+(np.interp(np.floor(u*10)/10,t,feed)-baseline)*4096/50.27).astype(int)
        for j,sec in enumerate(u):
            if sec>=hi: break
            if (first+j)%147457 in range(90000,90004): continue
            us = (tick+round((sec-lo)*1e6)) % 2**32
            sequence = seq+first+j+1
            a,b,c = counts[j]
            f = int(feeds[j]); v = round(float(temps[j])*4)/4
            s = float(start+sec)
            received = s+.001
            record = dict(line=f"{us},{sequence},{a},{b},{c},{f},{v:.2f}",device_tick=us,tick_unit="us",
                          invalidates_segment=None,missing_packets_before=4 if (first+j)%147457==90004 else 0,
                          sequence=sequence,segment=f"mock-{n}-{lo:.6f}",sync="receipt_offset_unverified",
                          counts=[a,b,c],filament_counts=f,accel_m_s2=[a*9.81/256-9.81,b*9.81/256,c*9.81/256],
                          displacement=f*50.27/4096,temperature=v,receipt_minus_mapped_s=received-s)
            yield s,received,"sensor",record,None


def generate(n, params):
    name = f"run_{n:04d}"
    src = REAL/name
    api = list(original(src,"api"))
    rng = np.random.default_rng([params["seed"],n])
    fail = n in {2,3,4,5,6,9} or n>=22
    labels = dict(nozzle_health="failing" if fail else "healthy",feeder_health="healthy",motion_system_health="healthy")
    if src.exists():
        with (src/"features/api_features.csv").open() as f: start = timestamp(next(csv.DictReader(f))["window_start_utc"])
        length = 1200 if n==7 else api[-1][0]-start
    else:
        start = 1800000000+n*86400
        length = int(rng.integers(params["min_seconds"],params["max_seconds"]+1))
    severity = rng.uniform(.4,params["max_thermal_excursion"]) if fail else rng.uniform(0,1.3)
    state = motion(n,length,rng,api,start,severity)
    intervals = []
    def additions(lo,hi,base=None):
        if hi>lo:
            intervals.append([utc(start+lo),utc(start+hi)])
            yield from synth_sensor(lo,hi,start,state,rng,params,n,base)
    def sensors():
        prev = None
        last_valid = None
        for row in original(src,"sensor"):
            lo = 0 if prev is None else prev[0]-start+1/params["packet_hz"]
            if row[0]-start-lo>.5:
                yield from additions(lo,row[0]-start-.002,last_valid)
            yield row
            prev=row
            if row[3].get("counts") is not None:
                last_valid=row[3]
        yield from additions(prev[0]-start+.002 if prev else 0,length,last_valid)
    apis = api_stream(api,state,start,length,n,rng) if not api or n==7 else iter(api)
    counts = dict(api=0,sensor=0)
    cfg_values=json.loads((src/"settings.json").read_text()) if src.exists() else {}
    cfg_values.pop("feature_version",None)
    cfg=Config(**cfg_values)
    clock=time.perf_counter()
    dest=MOCK/name
    if dest.exists(): raise FileExistsError(dest)
    with tempfile.TemporaryDirectory(dir=MOCK) as tmp:
        staging=Path(tmp)
        if n>1: (staging/f"run_{n-1:04d}").mkdir()
        with Store(staging,labels,cfg) as store:
            scheduler=Scheduler(start,cfg)
            for s,receipt,source,r,row in heapq.merge(apis,sensors(),key=lambda item:item[0]):
                while item:=scheduler.take(s):
                    write_window(store,item,cfg,lambda _:None)
                scheduler.add(source,s,receipt,r)
                store.writers[source+"_raw"].writerow(row or dict(run_id=name,timestamp_utc=utc(s),receipt_utc=utc(receipt),record_json=json.dumps(r,separators=(",",":"))))
                counts[source]+=1
            while item:=scheduler.take(start+length,True):
                write_window(store,item,cfg,lambda _:None)
        stats=dict(run=name,origin="hybrid" if src.exists() else "synthetic",labels=labels,duration=length,
                   split="test" if n in TEST else "train_validation",raw=counts,windows=scheduler.index,
                   synthetic_sensor_intervals=intervals,original_api_records=len(api),seconds=time.perf_counter()-clock)
        store.path.rename(dest)
    (META/(name+".json")).write_text(json.dumps(stats,indent=2))
    print(name,counts,scheduler.index,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("runs",nargs="*",type=int)
    args=parser.parse_args()
    params=json.loads((META/"config.json").read_text())
    if MOCK.resolve() != DATA.resolve() / "mock_runs" or META.resolve() != DATA.resolve() / "mock_metadata":
        raise ValueError("mock output roots must be distinct local directories")
    MOCK.mkdir(exist_ok=True)
    for n in args.runs or range(1,31):
        if not 1<=n<=30: parser.error("run number outside 1..30")
        if (MOCK/f"run_{n:04d}").exists() and (META/f"run_{n:04d}.json").exists(): continue
        generate(n,params)

