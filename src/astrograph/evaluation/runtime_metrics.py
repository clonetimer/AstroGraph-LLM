from __future__ import annotations

import csv
import io
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def metrics_url_from_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return f"{base}/metrics"


def parse_prometheus_metrics(text: str) -> dict[str, list[dict[str, Any]]]:
    metrics: dict[str, list[dict[str, Any]]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            left, raw_value = line.rsplit(None, 1)
            value = float(raw_value)
        except ValueError:
            continue
        labels: dict[str, str] = {}
        if "{" in left and left.endswith("}"):
            name, raw_labels = left.split("{", 1)
            raw_labels = raw_labels[:-1]
            reader = csv.reader(io.StringIO(raw_labels), skipinitialspace=True)
            parts = next(reader, [])
            for part in parts:
                if "=" not in part:
                    continue
                key, item = part.split("=", 1)
                labels[key.strip()] = item.strip().strip('"')
        else:
            name = left
        metrics.setdefault(name, []).append({"labels": labels, "value": value})
    return metrics


def fetch_vllm_metrics(base_url: str, *, timeout_s: float = 3.0) -> dict[str, Any]:
    url = metrics_url_from_base_url(base_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            text = response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"available": False, "url": url, "error": str(exc), "metrics": {}}
    parsed = parse_prometheus_metrics(text)
    prefixes = (
        "vllm:num_requests_",
        "vllm:gpu_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:request_success_total",
        "vllm:e2e_request_latency_seconds",
        "vllm:time_to_first_token_seconds",
        "vllm:time_per_output_token_seconds",
    )
    selected = {name: values for name, values in parsed.items() if name.startswith(prefixes)}
    return {"available": True, "url": url, "error": None, "metrics": selected}


def collect_nvidia_smi() -> dict[str, Any]:
    fields = [
        "index", "name", "uuid", "memory.total", "memory.used", "memory.free",
        "utilization.gpu", "utilization.memory", "temperature.gpu", "power.draw",
    ]
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": str(exc), "gpus": []}
    rows = []
    for values in csv.reader(io.StringIO(completed.stdout)):
        values = [value.strip() for value in values]
        if len(values) != len(fields):
            continue
        row = dict(zip(fields, values))
        for key in ("index", "memory.total", "memory.used", "memory.free", "utilization.gpu", "utilization.memory", "temperature.gpu"):
            try:
                row[key] = int(float(row[key]))
            except ValueError:
                pass
        try:
            row["power.draw"] = float(row["power.draw"])
        except ValueError:
            pass
        rows.append(row)
    return {"available": True, "error": None, "gpus": rows}


def _metric_total(sample: dict[str, Any], metric_name: str) -> float | None:
    values = sample.get("vllm", {}).get("metrics", {}).get(metric_name)
    if not isinstance(values, list):
        return None
    return sum(float(item.get("value", 0.0)) for item in values)


def collect_runtime_metrics(
    *,
    base_url: str,
    duration_s: float = 0.0,
    interval_s: float = 1.0,
) -> dict[str, Any]:
    started = time.time()
    samples = []
    while True:
        samples.append({
            "timestamp_unix_s": time.time(),
            "nvidia_smi": collect_nvidia_smi(),
            "vllm": fetch_vllm_metrics(base_url),
        })
        elapsed = time.time() - started
        if elapsed >= duration_s:
            break
        time.sleep(min(interval_s, max(0.0, duration_s - elapsed)))
    derived: dict[str, Any] = {}
    if len(samples) >= 2:
        delta_s = samples[-1]["timestamp_unix_s"] - samples[0]["timestamp_unix_s"]
        for metric, key in (
            ("vllm:prompt_tokens_total", "prompt_tokens_per_s"),
            ("vllm:generation_tokens_total", "generation_tokens_per_s"),
        ):
            first = _metric_total(samples[0], metric)
            last = _metric_total(samples[-1], metric)
            if first is not None and last is not None and delta_s > 0:
                derived[key] = round((last - first) / delta_s, 6)
    return {
        "schema_version": "astrograph-runtime-metrics.v1",
        "base_url": base_url,
        "duration_s": round(time.time() - started, 6),
        "sample_count": len(samples),
        "derived": derived,
        "samples": samples,
    }


def write_runtime_metrics(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
