"""R21 fail-closed development admission shared by fitting and selection APIs."""
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
import json

from .external_dataset_contract import seal, verify_seal

REGISTRY = Path(__file__).resolve().parents[3] / "configs/evaluation/r21_data_exposure_registry.v1.json"
_scope = ContextVar("r21_dataset_use", default=None)
_accesses = ContextVar("r21_development_accesses", default=None)
BLOCK = "R21_SEEN_EVALUATION_REUSE_FOR_DEVELOPMENT_BLOCKED"


def registry():
    result = json.loads(REGISTRY.read_text(encoding="utf-8"))
    verify_seal(result)
    return result


def admit_dataset_use(dataset_id, partition, purpose):
    r = registry()
    dataset_id = r.get("aliases", {}).get(dataset_id, dataset_id)
    if purpose not in r["development_purposes"] + r["read_only_purposes"]:
        raise ValueError("R21_PURPOSE_UNREGISTERED")
    entry = r["datasets"].get(dataset_id)
    if entry is None or partition not in entry["partitions"]:
        raise ValueError("R21_DATASET_OR_PARTITION_UNREGISTERED")
    status = entry["partitions"][partition]
    if status == "FUTURE_HOLDOUT_RESERVED":
        raise ValueError("R21_FUTURE_HOLDOUT_ACCESS_BLOCKED")
    if status == "SEEN_EVALUATION_READ_ONLY" and purpose not in r["read_only_purposes"]:
        raise ValueError(BLOCK)
    return {"dataset_id": dataset_id, "partition": partition, "purpose": purpose,
            "exposure_status": status, "registry_sha256": r["receipt_sha256"]}


@contextmanager
def dataset_use(dataset_id, partition, purpose, *, group_ids=None):
    admission = admit_dataset_use(dataset_id, partition, purpose)
    groups = None if group_ids is None else sorted(set(map(str, group_ids)))
    entry = registry()["datasets"][admission["dataset_id"]]
    if groups is not None and admission["dataset_id"] != "SYNTHETIC_PROTOCOL":
        allowed = set(entry["group_ids_by_partition"].get(partition, []))
        if not set(groups) <= allowed:
            raise ValueError(BLOCK + ":GROUP_PROVENANCE_MISMATCH")
    token = _scope.set({**admission, "group_ids": groups})
    try:
        yield _scope.get()
    finally:
        _scope.reset(token)


def guard_development(purpose, *, group_ids=None):
    scope = _scope.get()
    if scope is None:
        raise ValueError("R21_DEVELOPMENT_PROVENANCE_REQUIRED")
    admission = admit_dataset_use(scope["dataset_id"], scope["partition"], purpose)
    groups = scope["group_ids"] if group_ids is None else sorted(set(map(str, group_ids)))
    if scope["dataset_id"] == "SYNTHETIC_PROTOCOL" and groups:
        real_groups = {g for name, entry in registry()["datasets"].items()
                       if name != "SYNTHETIC_PROTOCOL"
                       for ids in entry["group_ids_by_partition"].values() for g in ids}
        if set(groups) & real_groups:
            raise ValueError("R21_REAL_GROUPS_CANNOT_CLAIM_SYNTHETIC_PROVENANCE")
    if scope["dataset_id"] != "SYNTHETIC_PROTOCOL":
        if not groups or scope["group_ids"] is None:
            raise ValueError("R21_INPUT_GROUP_PROVENANCE_REQUIRED")
        allowed = set(registry()["datasets"][scope["dataset_id"]]["group_ids_by_partition"].get(scope["partition"], []))
        if not set(groups) <= allowed or not set(groups) <= set(scope["group_ids"]):
            raise ValueError(BLOCK + ":INPUT_GROUP_REUSE")
    access = {**admission, "group_ids": groups or []}
    journal = _accesses.get()
    if journal is not None:
        journal.append(access)
    return access



def guard_in_memory_development(
    purpose,
    *,
    group_ids=None,
):
    if _scope.get() is not None:
        return guard_development(
            purpose,
            group_ids=group_ids,
        )

    groups = sorted(set(map(str, group_ids or [])))
    r = registry()
    group_statuses: dict[str, set[str]] = {}
    for name, entry in r["datasets"].items():
        if name == "SYNTHETIC_PROTOCOL":
            continue
        for partition, ids in entry["group_ids_by_partition"].items():
            status = entry["partitions"][partition]
            for group_id in ids:
                group_statuses.setdefault(str(group_id), set()).add(str(status))

    blocked = {
        group_id: statuses
        for group_id in groups
        if (statuses := group_statuses.get(group_id))
        and any(status != "DEVELOPMENT_PERMITTED" for status in statuses)
    }
    if blocked:
        raise ValueError(
            "R21_DEVELOPMENT_PROVENANCE_REQUIRED:"
            "REGISTERED_NONDEVELOPMENT_GROUP"
        )

    registered_development = sorted(
        group_id for group_id in groups if group_id in group_statuses
    )
    access = {
        "dataset_id": (
            "UNSCOPED_REGISTERED_DEVELOPMENT_GROUPS"
            if registered_development
            else "UNSCOPED_IN_MEMORY_SYNTHETIC_ONLY"
        ),
        "partition": "IN_MEMORY",
        "purpose": purpose,
        "exposure_status": (
            "REGISTERED_DEVELOPMENT_GROUPS_COMPATIBILITY"
            if registered_development
            else "SYNTHETIC_ONLY_COMPATIBILITY"
        ),
        "group_ids": groups,
        "registry_sha256": r["receipt_sha256"],
    }
    journal = _accesses.get()
    if journal is not None:
        journal.append(access)
    return access

def guard_prediction_generation():
    scope = _scope.get()
    if scope is None:
        raise ValueError("R21_PREDICTION_INPUT_PROVENANCE_REQUIRED")
    if scope["exposure_status"] != "DEVELOPMENT_PERMITTED":
        raise ValueError("R21_SEEN_EVALUATION_PREDICTION_REGENERATION_BLOCKED")
    return guard_development("OPTIMIZE")


@contextmanager
def capture_development_accesses():
    entries = []
    token = _accesses.set(entries)
    try:
        yield entries
    finally:
        _accesses.reset(token)


def contamination_audit(development_receipts, preregistration):
    r = registry()
    violations = []
    blocked_hashes = {x["receipt_sha256"] for name, x in r["datasets"].items()
                      if name in ["DXC09_COMPETITION", "DXC09_REMATCH", "OPS_SAT", "ESA_MISSION1"]}
    locked = {name: {g for p, ids in x["group_ids_by_partition"].items()
                     if x["partitions"][p] == "SEEN_EVALUATION_READ_ONLY" for g in ids}
              for name, x in r["datasets"].items()}

    def scan(value, path, dataset=None, config=False):
        if isinstance(value, dict):
            current = value.get("dataset_id", dataset)
            current = r.get("aliases", {}).get(current, current)
            if current is not None and current not in r["datasets"]:
                violations.append({"path": path, "reason": "UNREGISTERED_DATASET_ID"})
            if (value.get("external_predictions_generated") or value.get("status") == "EXTERNAL_VALIDATED"
                    or str(value.get("schema_version", "")).startswith("astrograph-external-validation-receipt")):
                violations.append({"path": path, "reason": "R21_NEW_EXTERNAL_PREDICTION_OR_QUALIFICATION_FORBIDDEN"})
            if current in r["datasets"] and "partition" in value:
                try:
                    admit_dataset_use(current, value["partition"], value.get("purpose", "FIT"))
                except ValueError as exc:
                    violations.append({"path": path, "reason": str(exc)})
            for key, item in value.items():
                if config and key == "forbidden_receipt_hashes":
                    continue
                forbidden = locked.get(current, set()) if current else set().union(*locked.values())
                if key in forbidden:
                    violations.append({"path": path + "." + key, "reason": "LOCKED_GROUP_KEY_CONSUMED"})
                if key in ["group_ids", "training_ids", "selection_ids", "consumed_group_ids"] and isinstance(item, list):
                    overlap = sorted(set(map(str, item)) & forbidden)
                    if overlap:
                        violations.append({"path": path + "." + key, "reason": "LOCKED_GROUPS_CONSUMED", "groups": overlap})
                scan(item, path + "." + key, current, config)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                scan(item, path + f"[{index}]", dataset, config)
        elif isinstance(value, str):
            if value in blocked_hashes:
                violations.append({"path": path, "reason": "SEEN_EVALUATION_RECEIPT_CONSUMED"})
            forbidden = locked.get(dataset, set()) if dataset else set().union(*locked.values())
            if value in forbidden:
                violations.append({"path": path, "reason": "LOCKED_GROUP_VALUE_CONSUMED"})

    frozen_path = REGISTRY.with_name("r21_remediation_preregistration.v1.json")
    frozen_plan = json.loads(frozen_path.read_text(encoding="utf-8"))
    verify_seal(frozen_plan)
    if preregistration != frozen_plan:
        violations.append({"path": "preregistration", "reason": "PREREGISTRATION_CHANGED"})
    for family in ["OPS", "ESA", "DXC"]:
        for declared in preregistration[family]["allowed_development"]:
            try:
                admit_dataset_use(declared["dataset_id"], declared["partition"], "FIT")
            except ValueError as exc:
                violations.append({"path": family + ".allowed_development", "reason": str(exc)})
    scan(preregistration, "preregistration", config=True)
    for index, receipt in enumerate(development_receipts):
        scan(receipt, f"receipts[{index}]")
    return seal({"schema_version": "astrograph-r21-contamination-audit.v1",
                 "status": "INVALID_SEEN_EVALUATION_CONTAMINATION" if violations else "PASS",
                 "violations": violations, "development_receipt_count": len(development_receipts),
                 "registry_sha256": r["receipt_sha256"],
                 "preregistration_sha256": frozen_plan["receipt_sha256"]})
