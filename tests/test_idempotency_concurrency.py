from __future__ import annotations

import json
import threading
from pathlib import Path

from astrograph.idempotency import JsonIdempotencyStore


def test_different_idempotency_keys_run_producers_concurrently(tmp_path: Path) -> None:
    store = JsonIdempotencyStore(tmp_path / "requests.json")
    both_entered = threading.Barrier(3)
    release = threading.Event()
    results: list[tuple[dict, bool]] = []
    errors: list[BaseException] = []

    def worker(key: str) -> None:
        try:
            def producer() -> dict:
                both_entered.wait(timeout=2)
                release.wait(timeout=2)
                return {"key": key}

            results.append(store.execute(key, {"key": key}, producer))
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(key,)) for key in ("A", "B")]
    for thread in threads:
        thread.start()
    both_entered.wait(timeout=2)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert errors == []
    assert len(results) == 2
    assert all(replayed is False for _, replayed in results)
    payload = json.loads((tmp_path / "requests.json").read_text(encoding="utf-8"))
    assert set(payload["records"]) == {"A", "B"}


def test_same_key_executes_producer_once_and_replays(tmp_path: Path) -> None:
    store = JsonIdempotencyStore(tmp_path / "requests.json")
    producer_started = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    producer_count = 0
    results: list[tuple[dict, bool]] = []

    def producer() -> dict:
        nonlocal producer_count
        with count_lock:
            producer_count += 1
        producer_started.set()
        release.wait(timeout=2)
        return {"value": 7}

    def worker() -> None:
        results.append(store.execute("same", {"request": 1}, producer))

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    producer_started.wait(timeout=2)
    second.start()
    release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert producer_count == 1
    assert sorted(replayed for _, replayed in results) == [False, True]
    assert all(response == {"value": 7} for response, _ in results)
    assert list(tmp_path.glob(".*.tmp")) == []
