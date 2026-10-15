from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

from astrograph.idempotency import SqliteIdempotencyStore


def _process_worker(db_path: str, marker_path: str, queue) -> None:
    store = SqliteIdempotencyStore(db_path, wait_timeout_s=5.0, poll_interval_s=0.01)

    def producer() -> dict:
        with open(marker_path, "a", encoding="utf-8") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        time.sleep(0.2)
        return {"value": 9}

    queue.put(store.execute("shared", {"request": 1}, producer))


def test_sqlite_store_runs_different_keys_concurrently(tmp_path: Path) -> None:
    store = SqliteIdempotencyStore(tmp_path / "requests.sqlite3")
    entered = threading.Barrier(3)
    release = threading.Event()
    results: list[tuple[dict, bool]] = []

    def worker(key: str) -> None:
        def producer() -> dict:
            entered.wait(timeout=2)
            release.wait(timeout=2)
            return {"key": key}

        results.append(store.execute(key, {"key": key}, producer))

    threads = [threading.Thread(target=worker, args=(key,)) for key in ("A", "B")]
    for thread in threads:
        thread.start()
    entered.wait(timeout=2)
    release.set()
    for thread in threads:
        thread.join(timeout=2)

    assert len(results) == 2
    assert all(replayed is False for _, replayed in results)


def test_sqlite_store_prevents_duplicate_same_key_across_processes(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    db_path = tmp_path / "requests.sqlite3"
    marker = tmp_path / "producer_calls.txt"
    queue = context.Queue()
    processes = [
        context.Process(target=_process_worker, args=(str(db_path), str(marker), queue))
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=10) for _ in processes]
        for process in processes:
            process.join(timeout=10)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            process.close()
        queue.close()
        queue.join_thread()

    assert marker.read_text(encoding="utf-8").splitlines().__len__() == 1
    assert sorted(replayed for _, replayed in results) == [False, True]
    assert all(response == {"value": 9} for response, _ in results)

def _init_worker(db_path: str, queue) -> None:
    SqliteIdempotencyStore(db_path, wait_timeout_s=5.0, poll_interval_s=0.01)
    queue.put("READY")


def test_sqlite_store_concurrent_initialization_is_retry_safe(tmp_path: Path) -> None:
    context = mp.get_context("spawn")
    db_path = tmp_path / "concurrent_init.sqlite3"
    queue = context.Queue()
    processes = [context.Process(target=_init_worker, args=(str(db_path), queue)) for _ in range(6)]
    for process in processes:
        process.start()
    try:
        results = [queue.get(timeout=15) for _ in processes]
        for process in processes:
            process.join(timeout=15)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            process.close()
        queue.close()
        queue.join_thread()
    assert results == ["READY"] * len(processes)



def test_sqlite_store_renews_long_running_lease_and_prevents_stale_reclaim(tmp_path: Path) -> None:
    db_path = tmp_path / "heartbeat.sqlite3"
    stores = [
        SqliteIdempotencyStore(
            db_path,
            wait_timeout_s=3.0,
            poll_interval_s=0.01,
            stale_after_s=0.15,
            heartbeat_interval_s=0.03,
        )
        for _ in range(2)
    ]
    producer_started = threading.Event()
    producer_count = 0
    count_lock = threading.Lock()
    results: list[tuple[dict, bool]] = []
    errors: list[BaseException] = []

    def producer() -> dict:
        nonlocal producer_count
        with count_lock:
            producer_count += 1
        producer_started.set()
        time.sleep(0.45)  # three times stale_after_s; heartbeat must retain ownership
        return {"value": 11}

    def worker(store: SqliteIdempotencyStore) -> None:
        try:
            results.append(store.execute("long-running", {"request": 1}, producer))
        except BaseException as exc:  # pragma: no cover - assertion aid
            errors.append(exc)

    first = threading.Thread(target=worker, args=(stores[0],))
    second = threading.Thread(target=worker, args=(stores[1],))
    first.start()
    assert producer_started.wait(timeout=1.0)
    time.sleep(0.22)  # old implementation would now reclaim the supposedly stale row
    second.start()
    first.join(timeout=3.0)
    second.join(timeout=3.0)

    assert errors == []
    assert producer_count == 1
    assert sorted(replayed for _, replayed in results) == [False, True]
    assert all(response == {"value": 11} for response, _ in results)


def test_sqlite_store_migrates_pre_heartbeat_schema(tmp_path: Path) -> None:
    import sqlite3

    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE idempotency_records (
                idempotency_key TEXT PRIMARY KEY,
                request_hash TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('RUNNING', 'COMPLETED')),
                owner_token TEXT NOT NULL,
                response_hash TEXT,
                response_json TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.commit()

    SqliteIdempotencyStore(db_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(idempotency_records)")}
    assert "lease_epoch" in columns
