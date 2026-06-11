"""Connectivity detection.

These checks decide whether the assistant can use the internet and whether a
local Ollama server is running. They are intentionally fast and fail quiet.
"""

import socket
import urllib.request
from functools import lru_cache


def is_online(timeout: float = 1.5) -> bool:
    """Return True if the machine appears to have internet access.

    Tries a DNS-port TCP connection to a couple of well-known IPs. This avoids
    a full HTTP request and works even when DNS is flaky.
    """
    hosts = [("1.1.1.1", 53), ("8.8.8.8", 53)]
    for host, port in hosts:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def ollama_available(host: str, timeout: float = 1.5) -> bool:
    """Return True if a local Ollama server responds on the given host."""
    url = f"{host.rstrip('/')}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


@lru_cache(maxsize=1)
def cached_online_status() -> bool:
    """Cached online check for use within a single short-lived operation."""
    return is_online()
