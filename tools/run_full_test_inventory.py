from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / ".r36_validation"
LOGS = OUT / "logs"
RESULT_FILE = OUT / "results.tsv"


def _load_existing() -> dict[str, tuple[str, int, float, str]]:
    if not RESULT_FILE.is_file():
        return {}
    rows: dict[str, tuple[str, int, float, str]] = {}
    with RESULT_FILE.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            path = str(row.get("test_file") or "").strip()
            if not path:
                continue
            rows[path] = (
                str(row.get("status") or ""),
                int(row.get("return_code") or 0),
                float(row.get("seconds") or 0.0),
                path,
            )
    return rows


def _write_results(rows: list[tuple[str, int, float, str]]) -> None:
    temporary = RESULT_FILE.with_suffix(".tsv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["status", "return_code", "seconds", "test_file"])
        writer.writerows(rows)
    temporary.replace(RESULT_FILE)


def _run_one(test_file: Path, env: dict[str, str], timeout_s: int) -> tuple[str, int, float, str]:
    relative = test_file.relative_to(ROOT)
    logfile = LOGS / (str(relative).replace("/", "__").replace("\\", "__") + ".log")
    started = time.monotonic()
    # Write directly to a file instead of PIPE. Some tests spawn descendants that
    # inherit stdout; PIPE + communicate() can then wait for EOF after pytest exits.
    with logfile.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "pytest", "-q", str(relative)],
            cwd=ROOT,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            rc = int(process.wait(timeout=timeout_s))
            status = "PASS" if rc == 0 else "FAIL"
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=5)
                except Exception:
                    pass
            rc = 124
            status = "TIMEOUT"
    elapsed = round(time.monotonic() - started, 3)
    return status, rc, elapsed, str(relative)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run each pytest file in an isolated process and persist incremental R36 results.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-file timeout in seconds")
    parser.add_argument("--resume", action="store_true", help="Skip files already present in results.tsv")
    parser.add_argument("--max-files", type=int, default=None, help="Run at most this many pending files")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("timeout must be positive")
    if args.max_files is not None and args.max_files <= 0:
        raise ValueError("max-files must be positive")

    tests = sorted((ROOT / "tests").rglob("test_*.py"))
    OUT.mkdir(exist_ok=True)
    LOGS.mkdir(exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + str(ROOT)
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = "1"

    existing = _load_existing() if args.resume else {}
    by_path = dict(existing)
    pending = [item for item in tests if str(item.relative_to(ROOT)) not in existing]
    if args.max_files is not None:
        pending = pending[: args.max_files]

    for ordinal, test_file in enumerate(pending, 1):
        relative = str(test_file.relative_to(ROOT))
        print(f"[{ordinal}/{len(pending)} pending; {len(by_path)}/{len(tests)} recorded] {relative}", flush=True)
        row = _run_one(test_file, env, args.timeout)
        by_path[relative] = row
        ordered = [by_path[str(item.relative_to(ROOT))] for item in tests if str(item.relative_to(ROOT)) in by_path]
        _write_results(ordered)
        print(row[0], row[2], "s", flush=True)

    ordered = [by_path[str(item.relative_to(ROOT))] for item in tests if str(item.relative_to(ROOT)) in by_path]
    summary = {
        status: sum(row[0] == status for row in ordered)
        for status in ("PASS", "FAIL", "TIMEOUT")
    }
    summary["RECORDED"] = len(ordered)
    summary["TOTAL"] = len(tests)
    summary["PENDING"] = len(tests) - len(ordered)
    print(summary, flush=True)
    return 1 if summary["FAIL"] or summary["TIMEOUT"] or summary["PENDING"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
