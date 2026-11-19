"""Read-only source loading and deterministic, unnormalized CNN caches."""
import csv
import hashlib
import json
import math
from array import array
from pathlib import Path
import numpy as np
import pandas as pd
from models.random_forest.random_forest import load_run

ROOT=Path(__file__).resolve().parents[2]
FEATURES=["api_extruder_temperature_"+s for s in ("mean","std","slope","error")]+[
 "api_head_velocity_mean","api_head_velocity_std","filament_velocity_mean",
 "filament_movement_rate_std","filament_movement_fraction_pct","filament_longest_no_movement_interval_s"]+[
 "thermocouple_temperature_"+s for s in ("mean","std","slope")]
VERSION=2

def stamp(s):
    return pd.Timestamp(s).value/1e9

def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf8")

def source_root(dataset):
    if dataset not in ("mock","real"): raise ValueError("Explicit mock or real dataset required")
    return ROOT/"data"/("mock_runs" if dataset=="mock" else "runs")

def labels(dataset,names):
    result=[]
    for name in names:
        if Path(name).name!=name or not name.startswith("run_"): raise ValueError("Invalid run name")
        f=pd.read_csv(source_root(dataset)/name/"labels/print_labels.csv")
        if len(f)!=1 or f.iloc[0].nozzle_health not in ("healthy","failing"): raise ValueError("Invalid run label")
        result.append(int(f.iloc[0].nozzle_health=="failing"))
    return result

def resample(t,x,segments,start,end,cfg):
    """Return channel-first interpolation, rejecting unsupported or ambiguous data."""
    if not len(t): raise ValueError("no_acceleration")
    lo,hi=np.searchsorted(t,[start,end]);inside=t[lo:hi]
    if len(inside)<math.ceil(cfg["coverage"]*cfg["hz"]*10): raise ValueError("insufficient_count")
    if inside[-1]-inside[0]<cfg["min_span"]: raise ValueError("insufficient_span")
    if np.max(np.diff(inside))>cfg["max_gap"]: raise ValueError("internal_gap")
    grid=start+np.arange(round(10*cfg["hz"]))/cfg["hz"]
    right=np.searchsorted(t,grid);exact=(right<len(t)) & (t[np.minimum(right,len(t)-1)]==grid)
    left=np.where(exact,right,right-1)
    if np.any(left<0) or np.any(right>=len(t)): raise ValueError("unsupported_boundary")
    a,b=min(lo,int(left.min())),max(hi,int(right.max())+1)
    if len(set(segments[a:b]))!=1: raise ValueError("segment_boundary")
    if not np.isfinite(x[a:b]).all(): raise ValueError("invalid_acceleration")
    if np.any(t[right]-t[left]>cfg["max_gap"]): raise ValueError("interpolation_gap")
    return np.stack([np.interp(grid,t[a:b],x[a:b,j]) for j in range(3)]).astype("float32")

def prepare(dataset,name,cfg):
    root=source_root(dataset)/name
    labels(dataset,[name])
    paths=[root/p for p in ("raw/sensor_raw.csv","features/api_features.csv","features/sensor_features.csv","labels/print_labels.csv","settings.json")]
    digest=hashlib.sha256(json.dumps(dict(version=VERSION,config=cfg,features=FEATURES),sort_keys=True).encode())
    signatures={}
    for path in paths:
        h=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b""):h.update(chunk)
        signatures[str(path.relative_to(root))]=h.hexdigest();digest.update(h.digest())
    dest=ROOT/"data/cnn_cache"/dataset/digest.hexdigest()[:20]/name
    if (dest/"manifest.json").exists():
        manifest=json.loads((dest/"manifest.json").read_text())
        if manifest["fingerprint"]!=digest.hexdigest(): raise ValueError("Cache fingerprint collision")
        for filename in ("acceleration.npy","auxiliary.npy","windows.csv","eligibility.csv"):
            if not (dest/filename).is_file():raise ValueError("Incomplete cache")
        return dest
    if dest.exists(): raise ValueError(f"Incomplete cache: {dest}; choose a clean cache location")
    if cfg["hz"]<=0 or not 0<cfg["coverage"]<=1 or cfg["max_gap"]<=0 or not 0<cfg["min_span"]<=10 or abs(10*cfg["hz"]-round(10*cfg["hz"]))>1e-8:
        raise ValueError("Invalid preprocessing settings")
    settings=json.loads((root/"settings.json").read_text())
    if settings.get("feature_version")!=5: raise ValueError("Only verified feature version 5 is supported")
    frame,_,_=load_run(root)
    if not frame.run_id.eq(name).all(): raise ValueError("Run identifier mismatch")
    starts=np.array([stamp(s) for s in frame.window_start_utc]);ends=np.array([stamp(s) for s in frame.window_end_utc])
    if not np.allclose(ends-starts,10,rtol=0,atol=1e-6) or (len(starts)>1 and not np.allclose(np.diff(starts),2,rtol=0,atol=1e-6)):
        raise ValueError("Invalid existing window schedule")
    t=array('d');receipts=array('d');values=array('f');segments=[];invalidated=set();previous=-math.inf
    from datetime import datetime
    with paths[0].open(newline="") as f:
        for row in csv.DictReader(f):
            r=json.loads(row["record_json"])
            if row["run_id"]!=name:raise ValueError("Raw run identifier mismatch")
            if r.get("invalidates_segment") is not None:invalidated.add(str(r["invalidates_segment"]))
            if "error" in r or r.get("sync")=="out_of_order":continue
            ts=datetime.fromisoformat(row["timestamp_utc"].replace("Z","+00:00")).timestamp()
            if ts<previous:raise ValueError("Nonmonotonic raw timestamp")
            v=r.get("accel_m_s2",[math.nan]*3);seg=str(r.get("segment"))
            if len(v)!=3:raise ValueError("Invalid XYZ shape")
            if ts==previous:
                if list(values[-3:])!=np.array(v,dtype="float32").tolist() or segments[-1]!=seg:values[-3:]=array('f',[math.nan]*3)
                continue
            receipts.append(datetime.fromisoformat(row["receipt_utc"].replace("Z","+00:00")).timestamp());t.append(ts);values.extend([math.nan if a is None else a for a in v]);segments.append(seg);previous=ts
    receipts=np.asarray(receipts);t=np.asarray(t);x=np.asarray(values).reshape(-1,3);segments=np.array(segments)
    x[np.isin(segments,list(invalidated))]=np.nan
    dest.mkdir(parents=True)
    accel=np.lib.format.open_memmap(dest/"acceleration.npy",mode="w+",dtype="float32",shape=(len(frame),3,round(cfg["hz"]*10)))
    aux=np.lib.format.open_memmap(dest/"auxiliary.npy",mode="w+",dtype="float32",shape=(len(frame),13))
    accepted=[];report=[]
    for i,row in frame.iterrows():
        reasons=[];v=pd.to_numeric(row[FEATURES],errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(v).all():reasons.append("missing_auxiliary")
        try:
            lo,hi=np.searchsorted(t,[starts[i],ends[i]])
            if np.any(receipts[max(0,lo-1):min(len(t),hi+1)]>ends[i]+settings.get("allowance",.5)):
                raise ValueError("late_receipt")
            a=resample(t,x,segments,starts[i],ends[i],cfg)
        except ValueError as exc:reasons.append(str(exc))
        entry={k:row[k] for k in ("run_id","window_id","window_start_utc","window_end_utc")}
        entry["true"]=int(row.nozzle_health=="failing")
        report.append({**entry,"reasons":";".join(reasons)})
        if not reasons:
            j=len(accepted);accel[j]=a;aux[j]=v;accepted.append(entry)
    accel.flush();aux.flush();del accel,aux
    pd.DataFrame(accepted,columns=["run_id","window_id","window_start_utc","window_end_utc","true"]).to_csv(dest/"windows.csv",index=False)
    pd.DataFrame(report).to_csv(dest/"eligibility.csv",index=False)
    dump(dest/"manifest.json",dict(dataset=dataset,run=name,source=str(root),fingerprint=digest.hexdigest(),source_hashes=signatures,preprocessing=cfg,version=VERSION,features=FEATURES,candidate=len(frame),eligible=len(accepted)))
    print(f"{name}: {len(accepted)}/{len(frame)} eligible",flush=True)
    return dest

def normalization(caches,limit=None):
    """Equal-run population moments over eligible window tensors, in chunks."""
    result={}
    for key in ("acceleration","auxiliary"):
        means=[];seconds=[]
        for path in caches:
            n=len(pd.read_csv(path/"windows.csv"));n=min(n,limit) if limit else n;a=np.load(path/(key+".npy"),mmap_mode="r")
            if not n:raise ValueError(f"No eligible windows: {path}")
            total=0;s=0.;ss=0.
            for begin in range(0,n,32):
                b=np.asarray(a[begin:min(begin+32,n)],dtype=np.float64)
                axes=(0,2) if key=="acceleration" else (0,)
                count=b.shape[0]*(b.shape[2] if key=="acceleration" else 1)
                s=s+b.sum(axis=axes);ss=ss+(b*b).sum(axis=axes);total+=count
            means.append(s/total);seconds.append(ss/total)
        mean=np.mean(means,axis=0);sd=np.sqrt(np.maximum(0,np.mean(seconds,axis=0)-mean*mean));constant=sd<1e-8
        if constant.any():print(f"Normalization warning: {key} near-constant indices {np.flatnonzero(constant).tolist()}",flush=True)
        result[key]=dict(mean=mean.tolist(),scale=np.where(constant,1,sd).tolist(),constant_indices=np.flatnonzero(constant).tolist())
    return result

class Windows:
    def __init__(self,caches,norm,limit=None):
        self.arrays=[];frames=[]
        for path in caches:
            f=pd.read_csv(path/"windows.csv")
            if limit:f=f.head(limit)
            if f.empty:raise ValueError(f"No eligible windows: {path}")
            f["local_index"]=np.arange(len(f));f["array_index"]=len(self.arrays)
            frames.append(f);self.arrays.append((np.load(path/"acceleration.npy",mmap_mode="r"),np.load(path/"auxiliary.npy",mmap_mode="r")))
        self.frame=pd.concat(frames,ignore_index=True);self.norm=norm
        counts=self.frame.run_id.value_counts();self.weights=np.array([len(self.frame)/(len(counts)*counts[r]) for r in self.frame.run_id],dtype="float32")
    def close(self):
        """Release mapped files explicitly, including Windows file handles."""
        for pair in self.arrays:
            for a in pair:a._mmap.close()
        self.arrays=[]
    def __len__(self):return len(self.frame)
    def __getitem__(self,i):
        r=self.frame.iloc[i];a,b=self.arrays[int(r.array_index)];j=int(r.local_index)
        an=self.norm["acceleration"];bn=self.norm["auxiliary"]
        x=(a[j]-np.array(an["mean"])[:,None])/np.array(an["scale"])[:,None]
        v=(b[j]-bn["mean"])/bn["scale"]
        return x.astype("float32"),v.astype("float32"),np.float32(r.true),self.weights[i]


def eligibility_summary(caches):
    """Summarize preparation losses without counting unused acceleration features."""
    from collections import Counter
    runs=[];classes={};reasons=Counter()
    for path in caches:
        f=pd.read_csv(path/"eligibility.csv",keep_default_na=False)
        accepted=f.reasons.eq("");label=int(f.true.iloc[0])
        counts=Counter(reason for value in f.reasons for reason in value.split(";") if reason)
        reasons.update(counts)
        row=dict(run=str(f.run_id.iloc[0]),label=label,candidate=len(f),eligible=int(accepted.sum()),rejected=int((~accepted).sum()),reasons=dict(counts))
        runs.append(row);group=classes.setdefault(str(label),dict(candidate=0,eligible=0))
        group["candidate"]+=len(f);group["eligible"]+=int(accepted.sum())
    return dict(runs=runs,classes=classes,reasons=dict(reasons),candidate=sum(r["candidate"] for r in runs),eligible=sum(r["eligible"] for r in runs))
