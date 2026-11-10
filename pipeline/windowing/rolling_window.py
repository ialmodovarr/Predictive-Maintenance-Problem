from collections import deque
from threading import Lock


class Scheduler:
    """One coordinator owns the index. Collectors can append concurrently."""
    def __init__(self, anchor, config):
        self.anchor = anchor
        self.config = config
        self.index = 0
        self.lock = Lock()
        self.buffers = {key:deque() for key in ("api","sensor")}
        self.dropped = dict.fromkeys(self.buffers,0)
        self.late = dict.fromkeys(self.buffers,0)
        self.invalidated = dict.fromkeys(self.buffers,0)

    def add(self,source,t,receipt,record):
        with self.lock:
            previous_segment = record.get("invalidates_segment")
            if previous_segment is not None:
                before = len(self.buffers[source])
                self.buffers[source] = deque(row for row in self.buffers[source]
                                             if row[2].get("segment") != previous_segment)
                self.invalidated[source] += before-len(self.buffers[source])
            if record.get("sync") == "out_of_order":
                self.invalidated[source] += 1
                return
            start = self.anchor+self.index*self.config.stride
            if t < start:
                self.late[source] += 1
                return
            buf = self.buffers[source]
            if len(buf)>=self.config.buffer_limit:
                buf.popleft()
                self.dropped[source] += 1
            buf.append((t,receipt,record))

    def take(self, now, stopping=False):
        with self.lock:
            start = self.anchor+self.index*self.config.stride
            end = start+self.config.window
            deadline = end+self.config.allowance
            if now < (end if stopping else deadline):
                return None
            data, diagnostics = {}, {}
            for name,buf in self.buffers.items():
                selected = [(t,r,rec) for t,r,rec in buf if start<=t<end]
                data[name] = [(t,rec) for t,r,rec in selected if r<=deadline]
                diagnostics[name] = {"raw_count":len(selected),"late_excluded":sum(r>deadline for _,r,_ in selected),
                                     "arrived_after_eviction_total":self.late[name],
                                     "buffer_evictions_total":self.dropped[name],
                                     "timing_excluded_total":self.invalidated[name],
                                     "shortened_wait":now<deadline}
                self.buffers[name] = deque(row for row in buf if row[0]>=start+self.config.stride)
            self.index += 1
            return self.index,start,end,data,diagnostics
