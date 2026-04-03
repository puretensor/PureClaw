"""Prometheus metrics — pure-Python text format exporter.

No external dependencies. Thread-safe via GIL for atomic dict updates.
Exposed via /metrics endpoint on the health probe HTTP server (port 9876).
"""

import threading
import time

_lock = threading.Lock()
_start_time = time.monotonic()

# Registries: name -> {labels_tuple: value}
_counters: dict[str, dict[tuple, float]] = {}
_gauges: dict[str, dict[tuple, float]] = {}
_histograms: dict[str, dict[tuple, tuple[float, int, dict[float, int]]]] = {}

# Metric metadata: name -> (type, help)
_meta: dict[str, tuple[str, str]] = {}

# Default histogram buckets (latency in seconds)
_DEFAULT_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float("inf"))


def _register(name: str, mtype: str, helptext: str = ""):
    """Register metric metadata (idempotent)."""
    if name not in _meta:
        _meta[name] = (mtype, helptext)


def _labels_key(labels: dict | None) -> tuple:
    """Convert labels dict to sorted tuple for use as dict key."""
    if not labels:
        return ()
    return tuple(sorted(labels.items()))


def inc(name: str, labels: dict | None = None, value: float = 1):
    """Increment a counter."""
    _register(name, "counter")
    key = _labels_key(labels)
    with _lock:
        bucket = _counters.setdefault(name, {})
        bucket[key] = bucket.get(key, 0) + value


def set_gauge(name: str, labels: dict | None = None, value: float = 0):
    """Set a gauge value."""
    _register(name, "gauge")
    key = _labels_key(labels)
    with _lock:
        bucket = _gauges.setdefault(name, {})
        bucket[key] = value


def observe(name: str, labels: dict | None = None, value: float = 0,
            buckets: tuple = _DEFAULT_BUCKETS):
    """Record a histogram observation."""
    _register(name, "histogram")
    key = _labels_key(labels)
    with _lock:
        hist_bucket = _histograms.setdefault(name, {})
        if key not in hist_bucket:
            hist_bucket[key] = (0.0, 0, {b: 0 for b in buckets})
        total, count, bounds = hist_bucket[key]
        total += value
        count += 1
        for b in bounds:
            if value <= b:
                bounds[b] += 1
        hist_bucket[key] = (total, count, bounds)


def _escape_label_value(v) -> str:
    """Escape label value per Prometheus exposition format."""
    s = str(v)
    s = s.replace("\\", "\\\\")
    s = s.replace('"', '\\"')
    s = s.replace("\n", "\\n")
    return s


def _format_labels(key: tuple) -> str:
    """Format labels tuple as Prometheus label string."""
    if not key:
        return ""
    parts = ",".join(f'{k}="{_escape_label_value(v)}"' for k, v in key)
    return "{" + parts + "}"


def render() -> str:
    """Render all metrics in Prometheus text exposition format."""
    lines = []

    # Built-in uptime gauge
    set_gauge("nexus_uptime_seconds", value=time.monotonic() - _start_time)

    with _lock:
        # Counters
        for name, bucket in sorted(_counters.items()):
            mtype, helptext = _meta.get(name, ("counter", ""))
            if helptext:
                lines.append(f"# HELP {name} {helptext}")
            lines.append(f"# TYPE {name} counter")
            for key, val in sorted(bucket.items()):
                lines.append(f"{name}{_format_labels(key)} {val}")

        # Gauges
        for name, bucket in sorted(_gauges.items()):
            mtype, helptext = _meta.get(name, ("gauge", ""))
            if helptext:
                lines.append(f"# HELP {name} {helptext}")
            lines.append(f"# TYPE {name} gauge")
            for key, val in sorted(bucket.items()):
                lines.append(f"{name}{_format_labels(key)} {val}")

        # Histograms (bounds are already cumulative from observe())
        for name, bucket in sorted(_histograms.items()):
            mtype, helptext = _meta.get(name, ("histogram", ""))
            if helptext:
                lines.append(f"# HELP {name} {helptext}")
            lines.append(f"# TYPE {name} histogram")
            for key, (total, count, bounds) in sorted(bucket.items()):
                labels_str = _format_labels(key)
                # Remove closing brace for adding le label
                base = labels_str.rstrip("}") if labels_str else "{"
                sep = "," if base != "{" else ""
                for le_val in sorted(b for b in bounds if b != float("inf")):
                    le_str = f"{le_val:g}"
                    lines.append(f'{name}_bucket{base}{sep}le="{le_str}"}} {bounds[le_val]}')
                # +Inf bucket
                lines.append(f'{name}_bucket{base}{sep}le="+Inf"}} {bounds.get(float("inf"), count)}')
                lines.append(f"{name}_sum{labels_str} {total}")
                lines.append(f"{name}_count{labels_str} {count}")

    lines.append("")
    return "\n".join(lines)
