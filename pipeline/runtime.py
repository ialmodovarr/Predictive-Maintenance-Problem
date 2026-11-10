import json
import time
from threading import Thread
from .collectors import api_collector, sensor_collector
from .features import api_features, sensor_features
from .storage.csv_writer import Store
from .synchronization.timestamp_sync import utc
from .windowing.rolling_window import Scheduler


def record(root, labels, config, stop, collectors=None):
    collectors = collectors or {"api":api_collector.collect,"sensor":sensor_collector.collect}
    with Store(root,labels,config) as store:
        anchor = time.time()
        scheduler = Scheduler(anchor,config)
        failures = []
        counts = {"api":0,"sensor":0}

        def worker(name,collect):
            def emit(t,received,row):
                try:
                    store.raw(name,t,received,row)
                    scheduler.add(name,t,received,row)
                    counts[name] += 1
                except Exception as exc:
                    failures.append(exc)
                    stop.set()
                    raise
            while not stop.is_set():
                try:
                    collect(config,emit,stop)
                    if not stop.is_set():
                        raise RuntimeError("collector disconnected")
                except (Exception,SystemExit) as exc:
                    print(f"{name} interrupted: {exc}",flush=True)
                    stop.wait(1)

        def drain(now,stopping=False):
            while item := scheduler.take(now,stopping):
                i,start,end,data,stats = item
                for name,module in (("api",api_features),("sensor",sensor_features)):
                    values,details = module.extract(data[name],config)
                    lag = max(0,time.time()-end)
                    store.append(name+"_features",{"window_id":f"window_{i:04d}",
                                 "window_start_utc":utc(start),"window_end_utc":utc(end),
                                 **store.labels,**values,"processing_delay_s":lag,
                                 "diagnostics":json.dumps({**details,**stats[name]})})
                print(f"{store.path.name} window {i} ended {utc(end)} delay={lag:.3f}s records={counts}",flush=True)

        threads = [Thread(target=worker,args=item,name=item[0]) for item in collectors.items()]
        print(f"Recording {store.path} {labels}",flush=True)
        for thread in threads:
            thread.start()
        try:
            while not stop.wait(.05):
                drain(time.time())
        finally:
            cutoff = time.time()
            stop.set()
            for thread in threads:
                thread.join()
            if not failures:
                drain(cutoff,True)
        if failures:
            raise RuntimeError("Recording failed") from failures[0]
        return store.path
