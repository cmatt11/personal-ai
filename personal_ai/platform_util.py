"""Cross-platform helpers so the assistant runs on Windows, macOS, Linux, and
Android (via Termux).

Avoids hard-coded Unix assumptions. Picks the right shell per OS and reports a
friendly platform summary.
"""

import os
import platform
import sys
from pathlib import Path


def os_name() -> str:
    """Return one of: windows, macos, linux, android, unknown."""
    plat = sys.platform
    # Termux on Android reports linux; detect it by its prefix.
    if "ANDROID_ROOT" in os.environ or "com.termux" in os.environ.get("PREFIX", ""):
        return "android"
    if plat.startswith("win"):
        return "windows"
    if plat == "darwin":
        return "macos"
    if plat.startswith("linux"):
        return "linux"
    return "unknown"


def is_windows() -> bool:
    return os_name() == "windows"


def shell_command(command: str):
    """Return the argv list to run a shell command on the current OS."""
    if is_windows():
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        return [comspec, "/c", command]
    return ["/bin/sh", "-c", command]


def default_home() -> Path:
    """A writable home directory that works across platforms, including Termux."""
    return Path(os.environ.get("HOME") or Path.home())


def summary() -> str:
    return (
        f"{os_name()} | python {platform.python_version()} | "
        f"{platform.machine()}"
    )
