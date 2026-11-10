import csv
import json
import re
from contextlib import ExitStack
from dataclasses import asdict
from pipeline.features import api_features, sensor_features
from pipeline.labels.label_manager import COMPONENTS
from pipeline.synchronization.timestamp_sync import utc


def allocate(root):
    root.mkdir(parents=True, exist_ok=True)
    ids = [int(p.name[4:]) for p in root.iterdir() if re.fullmatch(r"run_\d+",p.name)]
    i = max(ids,default=0)+1
    while True:
        path = root/f"run_{i:04d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            i += 1


class Store(ExitStack):
    def __init__(self, root, labels, config):
        super().__init__()
        if set(labels)!=set(COMPONENTS) or not set(labels.values())<={"healthy","failing"}:
            raise ValueError("Invalid component labels")
        self.labels = dict(labels)
        self.path = allocate(root)
        self.files = {}
        self.writers = {}
        raw = ("run_id","timestamp_utc","receipt_utc","record_json")
        meta = ("run_id","window_id","window_start_utc","window_end_utc",*COMPONENTS,"processing_delay_s","diagnostics")
        try:
            for name,folder,schema in (
                ("print_labels","labels",("run_id",*COMPONENTS)),
                ("api_raw","raw",raw),("sensor_raw","raw",raw),
                ("api_features","features",(*meta,*api_features.NAMES)),
                ("sensor_features","features",(*meta,*sensor_features.NAMES))):
                directory = self.path/folder
                directory.mkdir(exist_ok=True)
                f = self.enter_context((directory/(name+".csv")).open("x",newline="",encoding="utf8"))
                self.files[name] = f
                self.writers[name] = csv.DictWriter(f,fieldnames=schema)
                self.writers[name].writeheader()
                f.flush()
            self.append("print_labels",self.labels)
            (self.path/"settings.json").write_text(json.dumps({"feature_version":4,**asdict(config)},indent=2),encoding="utf8")
        except BaseException:
            self.close()
            raise

    def append(self,name,row):
        row = {"run_id":self.path.name,**row}
        writer = self.writers[name]
        if set(row)!=set(writer.fieldnames):
            raise ValueError(f"Unexpected schema for {name}")
        writer.writerow(row)
        self.files[name].flush()

    def raw(self,source,t,received,record):
        self.append(source+"_raw",{"timestamp_utc":utc(t),"receipt_utc":utc(received),
                                  "record_json":json.dumps(record,separators=(",",":"))})
