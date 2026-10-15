"""Step-1 data-availability audit for Sat-Agent schema profiles.

Reads ``configs/data_sources/sat_agent_platform_schema_profiles.s0.v1.json`` and
walks the Sat-Agent platform tree (read-only, pure file parsing) to decide, for
each profile, whether a real public telemetry export exists that matches the
profile's declared public observation fields and carries an explicit time axis.

Run with the AstroGraph interpreter only (uses stdlib + pandas, no imports of
AstroGraph or sat_agent_platform Python packages):

    /d/Project/test_lunwen/.venv/Scripts/python.exe scripts/audit_sat_agent_profiles.py \
        --sat-root /d/Project/test_sat/sat_agent_platform

It never writes to the Sat-Agent tree and never synthesizes time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Set of header tokens that identify a structural time axis (mirrors the
# adapter's ``_time_axis_candidates`` spirit).  ``sample_index`` is a row-index
# surrogate and is NOT treated as physical time.
_EXPLICIT_TIME_HINTS = {
    "time",
    "time_s",
    "timestamp",
    "datetime",
    "utc",
    "epoch",
}
_INDEX_SURROGATE_HINTS = {"index", "sample_index", "row_index"}


def _is_explicit_time(name: str) -> bool:
    lowered = name.strip().lower()
    if not lowered:
        return False
    if lowered in _INDEX_SURROGATE_HINTS:
        return False
    return (
        lowered in _EXPLICIT_TIME_HINTS
        or "timestamp" in lowered
        or "elapsed" in lowered
        or lowered.startswith("time_")
        or lowered.endswith("_time")
    )


def _csv_header(path: Path) -> list[str] | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [str(c).strip() for c in next(__import__("csv").reader(handle))]
    except (StopIteration, UnicodeDecodeError, OSError):
        return None


def _row_count(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return sum(1 for _ in handle) - 1
    except OSError:
        return 0


# Conservative signal that a column is a label / truth / identity / simulator
# answer proxy and must never enter a model feature set.
_FORBIDDEN_PREFIXES = (
    "label.",
    "label_",
    "truth.",
    "ground_truth.",
    "effect.",
    "event.",
    "modifier.",
    "object.",
    "status.",
    "task_id",
    "case_id",
    "dataset_id",
    "scenario_id",
    "pair_id",
    "run_id",
    "seed",
    "split",
    "mode",
    "target_level",
    "target_name",
    "sample_index",
)
_FORBIDDEN_CONTAINS = (
    "effective_",
    "effective.",
    ".effective_",
    ".bias",
    ".drift",
    ".degradation",
    ".fault",
    ".health",
    ".nominal",
    ".stuck",
    ".available",
    ".enabled",
    "fault_active",
    "degradation_active",
    "health_state",
)


def _has_forbidden_column(columns: list[str]) -> bool:
    for column in columns:
        lowered = column.strip().lower()
        if not lowered:
            continue
        if lowered.startswith(_FORBIDDEN_PREFIXES):
            return True
        if any(token in lowered for token in _FORBIDDEN_CONTAINS):
            return True
    return False


def _match_profile(columns: list[str], profile: dict) -> dict:
    public_fields = [item["name"] for item in profile["public_observation_fields"]]
    time_field = profile.get("time_field", {}).get("name", "time_s")
    present = [f for f in public_fields if f in columns]
    missing = [f for f in public_fields if f not in columns]
    explicit_time = [c for c in columns if _is_explicit_time(c)]
    has_declared_time = time_field in columns
    return {
        "public_field_count": len(public_fields),
        "present": present,
        "missing": missing,
        "explicit_time": explicit_time,
        "declared_time_field": time_field,
        "has_declared_time": has_declared_time,
        "full_match": len(missing) == 0 and has_declared_time,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sat-root",
        required=True,
        help="Absolute path to the Sat-Agent platform tree (read-only).",
    )
    parser.add_argument(
        "--profiles",
        default=Path(__file__).resolve().parents[2]
        / "configs"
        / "data_sources"
        / "sat_agent_platform_schema_profiles.s0.v1.json",
        help="Path to the AstroGraph schema-profile catalog.",
    )
    parser.add_argument("--limit", type=int, default=0, help="cap scanned files (debug)")
    args = parser.parse_args()

    sat_root = Path(args.sat_root).resolve()
    if not sat_root.is_dir():
        print(f"sat root missing: {sat_root}", file=sys.stderr)
        return 2

    profiles = json.loads(Path(args.profiles).read_text(encoding="utf-8"))["profiles"]
    print(f"profiles loaded: {len(profiles)}")
    print(f"sat root: {sat_root}\n")

    # Collect all CSV files, skipping noisy subtrees.
    exclude_parts = {".venv", "node_modules", "third_party", ".git"}
    csv_files: list[Path] = []
    for path in sat_root.rglob("*.csv"):
        if any(part in exclude_parts for part in path.parts):
            continue
        csv_files.append(path)
    if args.limit:
        csv_files = csv_files[: args.limit]
    print(f"csv files scanned: {len(csv_files)}\n")

    # header -> list of files sharing that exact header set
    headers: dict[tuple, list[Path]] = {}
    skipped = 0
    for path in csv_files:
        header = _csv_header(path)
        if header is None:
            skipped += 1
            continue
        headers.setdefault(tuple(header), []).append(path)
    print(f"distinct header variants: {len(headers)}; unreadable: {skipped}\n")

    rows = []
    for profile in profiles:
        pid = profile["profile_id"]
        matches = []
        for header, files in headers.items():
            result = _match_profile(list(header), profile)
            if result["full_match"]:
                candidates = [
                    {
                        "file": path,
                        "rows": _row_count(path),
                        "has_forbidden": _has_forbidden_column(list(header)),
                    }
                    for path in files
                ]
                candidates.sort(
                    key=lambda c: (c["has_forbidden"], -c["rows"])
                )
                best = candidates[0]
                matches.append(
                    {
                        "file": best["file"],
                        "rows": best["rows"],
                        "has_forbidden": best["has_forbidden"],
                        "variants": len(files),
                        "extra_columns": sorted(
                            set(header)
                            - set(result["present"])
                            - {t for t in result["explicit_time"]}
                        ),
                    }
                )
        matches.sort(key=lambda m: m["rows"], reverse=True)
        rows.append((profile, matches))

    print("=" * 100)
    for profile, matches in rows:
        pid = profile["profile_id"]
        cap = profile["capability_id"]
        fields = [f["name"] for f in profile["public_observation_fields"]]
        time_field = profile.get("time_field", {}).get("name", "time_s")
        print(f"\n[profile] {pid}")
        print(f"  capability      : {cap}")
        print(f"  declared time   : {time_field}")
        print(f"  declared fields : {fields}")
        if not matches:
            print("  MATCH  -> NONE on disk")
            continue
        print(f"  MATCH  -> {len(matches)} file(s) with ALL declared fields + explicit time")
        for m in matches[:4]:
            rel = m["file"].relative_to(sat_root)
            clean = "CLEAN" if not m["has_forbidden"] else "HAS-FORBIDDEN-COLUMNS"
            print(f"    - {rel}  (rows={m['rows']}, {clean}, header-variants={m['variants']})")
            if m["extra_columns"]:
                print(f"      extra columns present: {m['extra_columns'][:12]}{'...' if len(m['extra_columns']) > 12 else ''}")
        if len(matches) > 4:
            print(f"    ... and {len(matches) - 4} more")
    print("\n" + "=" * 100)

    # Summary availability matrix.
    print("\nAVAILABILITY MATRIX (profile | matched-file(s) | schema-match | time? | usable?)")
    for profile, matches in rows:
        pid = profile["profile_id"]
        if not matches:
            print(f"  {pid:55s} | NONE | no   | no    | NO")
            continue
        m0 = matches[0]
        usable = "YES" if m0["rows"] > 2 else "YES(short)"
        clean = "CLEAN" if not m0["has_forbidden"] else "FILTER-REQUIRED"
        print(
            f"  {pid:55s} | {m0['file'].relative_to(sat_root)} | yes  | yes   | {usable} ({clean})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
