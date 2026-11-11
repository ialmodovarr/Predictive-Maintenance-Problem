"""
Phone alerts for the dashboard through ntfy (https://ntfy.sh): free, no account.

People install the ntfy app and subscribe to a topic name. The dashboard publishes to
that topic and every subscribed phone gets a push notification. Anyone who knows the
topic name can subscribe, so it's a long random name rather than something guessable.

Topic: NTFY_TOPIC env var or .streamlit/secrets.toml, else a random one saved to
.streamlit/ntfy_topic.txt (gitignored) so it stays the same between runs on this laptop.
To share one topic across laptops, type the same topic into the sidebar on each.
"""
from __future__ import annotations

import functools
import os
import secrets
import threading
import time

import requests

TOPIC_FILE = os.path.join(".streamlit", "ntfy_topic.txt")
TIMEOUT_S = 6


def server(secrets_obj=None) -> str:
    return (_pick("NTFY_SERVER", secrets_obj) or "https://ntfy.sh").rstrip("/")


def load_topic(secrets_obj=None) -> str:
    topic = _pick("NTFY_TOPIC", secrets_obj)
    if topic:
        return topic
    try:
        with open(TOPIC_FILE) as f:
            topic = f.read().strip()
    except OSError:
        topic = ""
    if not topic:
        topic = "printer-health-" + secrets.token_hex(5)
        save_topic(topic)
    return topic


def save_topic(topic: str) -> None:
    try:
        os.makedirs(os.path.dirname(TOPIC_FILE), exist_ok=True)
        with open(TOPIC_FILE, "w") as f:
            f.write(topic.strip() + "\n")
    except OSError:
        pass                                   # read-only folder: topic just won't persist


def subscribe_url(srv: str, topic: str) -> str:
    return f"{srv}/{topic}"


def stamp(t: float | None = None) -> str:
    """'2:14:05 PM · Wed Nov 11' (built by hand: %-I doesn't work on Windows)."""
    lt = time.localtime(t)
    return (f"{lt.tm_hour % 12 or 12}:{lt.tm_min:02d}:{lt.tm_sec:02d} {'AM' if lt.tm_hour < 12 else 'PM'}"
            f" · {time.strftime('%a %b', lt)} {lt.tm_mday}")


@functools.lru_cache(maxsize=8)
def qr_svg(data: str) -> str:
    """QR code as an inline SVG string (no image library needed)."""
    import qrcode
    import qrcode.image.svg
    img = qrcode.make(data, image_factory=qrcode.image.svg.SvgPathImage, border=1)
    svg = img.to_string(encoding="unicode")
    return svg.replace("<svg ", "<svg class='qr' ", 1)


def send(srv: str, topic: str, title: str, message: str, priority: int = 3, tags=()) -> None:
    """Publish one notification. Raises on network or HTTP errors.

    JSON publishing (POST to the server root) so °C and other non-ASCII text survive.
    """
    r = requests.post(srv, json={"topic": topic, "title": title, "message": message,
                                 "priority": priority, "tags": list(tags)}, timeout=TIMEOUT_S)
    r.raise_for_status()


def send_async(srv: str, topic: str, title: str, message: str, status: dict, priority: int = 3, tags=()) -> None:
    """send() on a background thread so a slow network never stalls the dashboard refresh.

    status (a plain dict) gets "ok": bool and "error": str | None when it finishes.
    """
    def run():
        try:
            send(srv, topic, title, message, priority, tags)
            status.update(ok=True, error=None)
        except requests.RequestException as e:
            status.update(ok=False, error=str(e))
    threading.Thread(target=run, daemon=True).start()


def _pick(name, secrets_obj):
    if os.getenv(name):
        return os.getenv(name)
    try:
        return secrets_obj[name] if secrets_obj is not None and name in secrets_obj else ""
    except Exception:
        return ""
