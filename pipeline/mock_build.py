"""Run the two pilots, verify a deterministic rebuild, then build and audit the inventory."""
import concurrent.futures
import json
import shutil
import tempfile
import time
from pathlib import Path
from pipeline import mock_dataset as gen
from pipeline.mock_audit import audit,report


def build(n):
    gen.generate(n,json.loads((gen.META/'config.json').read_text()))
    return n


def finish():
    """Resume after a verified pilot build and audit before publication."""
    pilots=json.loads((gen.META/'pilots.json').read_text())
    if pilots.get('reproduced') != 8:
        raise ValueError("Missing pilot reproduction check")
    missing=[n for n in range(1,31) if not (gen.MOCK/f"run_{n:04d}").exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(build,missing))
    report()


def main():
    params=json.loads((gen.META/'config.json').read_text())
    pilots=[8,22]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(build,pilots))
        results=[audit(i) for i in pilots]
        for r in results:
            if r['complete']/r['windows']<.9: raise ValueError(r['run']+' insufficient complete windows')
        required=sum(r['bytes'] for r in results)/sum(r['duration'] for r in results)*36000
        free=shutil.disk_usage(gen.MOCK).free
        if free<required: raise OSError("insufficient free space")
        # Rebuild a full pilot in temporary isolation; compare all raw and feature bytes.
        import hashlib
        with tempfile.TemporaryDirectory(dir=gen.DATA) as tmp:
            root,meta=gen.MOCK,gen.META
            try:
                gen.MOCK=Path(tmp)/'runs';gen.META=Path(tmp)/'metadata'
                gen.MOCK.mkdir();gen.META.mkdir()
                gen.generate(8,params)
                for rel,sha in results[0]['sha256'].items():
                    with (gen.MOCK/'run_0008'/rel).open('rb') as f:
                        assert hashlib.file_digest(f,'sha256').hexdigest()==sha,rel
            finally: gen.MOCK,gen.META=root,meta
        (gen.META/'pilots.json').write_text(json.dumps(dict(pilots=results,estimated_full_bytes=required,free_bytes=free,reproduced=8),indent=2))
        list(pool.map(build,[n for n in range(1,31) if n not in pilots]))
    report()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--resume",action="store_true")
    if parser.parse_args().resume:
        finish()
    else:
        main()
