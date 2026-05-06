#!/usr/bin/env python3
"""Quick one-shot: connect to the MacBook and scroll down using the scroll skill."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "scroll"))

from daedalus.backends.vnc import VNCBackend

host = os.environ.get("DAEDALUS_VNC_HOST", "127.0.0.1")
port = int(os.environ.get("DAEDALUS_VNC_PORT", "5900"))
password = os.environ.get("DAEDALUS_VNC_PASSWORD")
username = os.environ.get("DAEDALUS_VNC_USERNAME")

backend = VNCBackend(
    host=host,
    port=port,
    password=password,
    username=username,
    max_resolution=(1728, 1117),
)

print(f"Connecting to {host}:{port}...")
backend.connect()
print(f"Connected. Screen size: {backend.size}")

# amount=10 means 10*5=50 VNC ticks, roughly half a page
print("Scrolling down 10 units (50 VNC ticks)...")
backend.scroll(dx=0, dy=10*50)
print("Done.")

backend.disconnect()
