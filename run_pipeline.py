"""Start both collectors and live tabular feature extraction."""
import argparse
from importlib.util import find_spec
import signal
from pathlib import Path
from threading import Event
from pipeline.config import Config
from pipeline.labels.label_manager import prompt_labels
from pipeline.runtime import record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",help="JSON file of pipeline.config.Config settings")
    args = parser.parse_args()
    if any(find_spec(module) is None for module in ("requests","serial")):
        parser.error("Install collector dependencies with python -m pip install -r requirements.txt")
    config = Config.load(args.config)
    if not config.printer_url.startswith(("http://","https://")):
        parser.error("Enter an HTTP or HTTPS printer URL")
    labels = prompt_labels()
    stop = Event()
    previous = signal.signal(signal.SIGINT,lambda *_:stop.set())
    try:
        record(Path(__file__).resolve().parent/"data"/"runs",labels,config,stop)
    finally:
        signal.signal(signal.SIGINT,previous)


if __name__ == "__main__":
    main()
