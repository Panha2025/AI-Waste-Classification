"""Capture transparent experiment metadata for professor's hardware requirement."""
from __future__ import annotations
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone


def hardware_report():
    d = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Not exposed by OS",
        "cpu_logical_cores": os.cpu_count(),
        "python": sys.version.split()[0],
    }
    try:
        import psutil
        d["ram_gib"] = round(psutil.virtual_memory().total / (1024**3), 2)
        d["cpu_physical_cores"] = psutil.cpu_count(logical=False)
    except ImportError:
        d["ram_gib"] = "psutil not installed"
    if sys.platform == "darwin":
        try:
            d["cpu_model"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, timeout=3).strip()
        except (FileNotFoundError, subprocess.SubprocessError):
            try:
                d["cpu_model"] = subprocess.check_output(["sysctl", "-n", "hw.model"], text=True, timeout=3).strip()
            except (FileNotFoundError, subprocess.SubprocessError):
                d["cpu_model"] = "Not available"
    try:
        import tensorflow as tf
        d["tensorflow_version"] = tf.__version__
        d["tensorflow_gpus"] = [str(x) for x in tf.config.list_physical_devices("GPU")]
        d["tensorflow_cpus"] = [str(x) for x in tf.config.list_physical_devices("CPU")]
    except ImportError:
        d["tensorflow_version"] = "Not installed"
        d["tensorflow_gpus"] = []
    return d
