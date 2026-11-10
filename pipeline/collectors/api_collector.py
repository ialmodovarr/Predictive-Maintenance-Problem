import time

from pipeline.config import PRINTER_USERNAME, PRINTER_PASSWORD


def collect(config, emit, stop):
    import requests
    from requests.auth import HTTPDigestAuth
    with requests.Session() as session:
        session.auth = HTTPDigestAuth(PRINTER_USERNAME, PRINTER_PASSWORD)
        while not stop.is_set():
            start = time.monotonic()
            for endpoint in ("/api/v1/print_job", "/api/v1/printer"):
                if stop.is_set():
                    break
                try:
                    response = session.get(config.printer_url.rstrip("/") + endpoint, timeout=config.timeout)
                    received = time.time()
                    response.raise_for_status()
                    payload = response.json()
                    emit(received, received, {"endpoint": endpoint, "payload": payload})
                except (requests.RequestException, ValueError) as exc:
                    print(f"API {endpoint}: {exc}", flush=True)
            stop.wait(max(0, config.poll - (time.monotonic()-start)))
