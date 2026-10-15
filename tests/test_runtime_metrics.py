from __future__ import annotations

from astrograph.evaluation.runtime_metrics import metrics_url_from_base_url, parse_prometheus_metrics


def test_metrics_url_removes_openai_v1_suffix():
    assert metrics_url_from_base_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/metrics"


def test_parse_prometheus_metrics_keeps_labels_and_values():
    parsed = parse_prometheus_metrics('''
# HELP ignored
vllm:num_requests_running{model_name="Qwen3.5-9B"} 2
vllm:prompt_tokens_total{model_name="Qwen3.5-9B"} 1234
''')
    assert parsed["vllm:num_requests_running"][0]["value"] == 2.0
    assert parsed["vllm:prompt_tokens_total"][0]["labels"]["model_name"] == "Qwen3.5-9B"
