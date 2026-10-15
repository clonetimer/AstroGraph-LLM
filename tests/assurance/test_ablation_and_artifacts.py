from __future__ import annotations

import json
import errno
import os
import stat
from pathlib import Path

import pytest

from astrograph.assurance import ArtifactPathError, ArtifactRoot
from astrograph.experiments import FrozenAblationPlan, FrozenAblationRunner
from astrograph.fusion import (
    BiRRFConfig,
    FaultRanking,
    KGFaultCoverageDeclaration,
    RankedFault,
    RankingSource,
    ScoreSemantics,
)


def _ranking(source: RankingSource) -> FaultRanking:
    semantics = {
        RankingSource.DATA_DRIVEN: ScoreSemantics.UNCALIBRATED_CLASS_PROBABILITY,
        RankingSource.KG_FORWARD: ScoreSemantics.EVIDENCE_STRENGTH,
        RankingSource.KG_BACKWARD: ScoreSemantics.EVIDENCE_COVERAGE,
    }[source]
    return FaultRanking(
        ranking_id=f"rank-{source.value.lower()}",
        episode_id="episode-1",
        source=source,
        score_semantics=semantics,
        candidates=[
            RankedFault(
                fault_id="F-A",
                rank=1,
                score=0.8,
                source=source,
                source_evidence_ids=["E-SHARED"],
                correlation_group_ids=["CORR-SHARED"],
                lineage_complete=True,
            ),
            RankedFault(
                fault_id="F-B",
                rank=2,
                score=0.2,
                source=source,
                source_evidence_ids=[f"E-{source.value}-B"],
                correlation_group_ids=["CORR-SHARED-B"],
                lineage_complete=True,
            ),
        ],
        producer_id=f"producer-{source.value.lower()}",
        producer_version="1",
        kg_fault_coverage=(
            None
            if source == RankingSource.DATA_DRIVEN
            else KGFaultCoverageDeclaration(
                source_artifact_id="test-evidence-package@1",
                source_artifact_sha256="a" * 64,
                supported_fault_ids=("F-A", "F-B"),
                evidence_definition_ids_by_fault={
                    "F-A": ("ED-F-A",),
                    "F-B": ("ED-F-B",),
                },
            )
        ),
    )


def _plan() -> FrozenAblationPlan:
    return FrozenAblationPlan(
        plan_id="frozen-plan-1",
        dataset_id="dataset-1",
        dataset_sha256="1" * 64,
        split_id="group-split-1",
        split_sha256="2" * 64,
        calibration_id="calibration-1",
        calibration_sha256="3" * 64,
        fault_universe=("F-A", "F-B"),
        policy_id="birrf-policy-1",
        policy_sha256="4" * 64,
        code_version="repair-b003",
        data_branch_id="data-v1",
        forward_branch_id="forward-v1",
        backward_branch_id="backward-v1",
        base_config=BiRRFConfig(
            collapse_identical_kg_rankings=False,
            collapse_highly_correlated_kg_rankings=False,
            collapse_shared_correlation_groups=True,
        ),
    )


def test_ablation_evaluates_each_branch_once_and_freezes_exact_outputs(tmp_path: Path) -> None:
    counts = {"data": 0, "forward": 0, "backward": 0}

    def factory(name: str, source: RankingSource):
        def build() -> FaultRanking:
            counts[name] += 1
            return _ranking(source)
        return build

    root = ArtifactRoot(tmp_path / "artifacts")
    bundle = FrozenAblationRunner(_plan(), artifact_root=root).run(
        data_factory=factory("data", RankingSource.DATA_DRIVEN),
        forward_factory=factory("forward", RankingSource.KG_FORWARD),
        backward_factory=factory("backward", RankingSource.KG_BACKWARD),
    )

    assert counts == {"data": 1, "forward": 1, "backward": 1}
    assert len(bundle.results) == 8
    assert len(bundle.receipts) == 8
    hashes = {
        tuple(sorted(receipt.branch_ranking_sha256.items()))
        for receipt in bundle.receipts.values()
    }
    assert len(hashes) == 1
    assert bundle.results["A6_ALL_UNADJUSTED"].candidates[0].fused_score > (
        bundle.results["A7_ALL_LINEAGE_ADJUSTED"].candidates[0].fused_score
    )
    assert bundle.results["A7_ALL_LINEAGE_ADJUSTED"].candidates[0].dependency_adjusted is True
    data_only = bundle.results["A0_DATA_ONLY"]
    assert all(
        set(candidate.source_ranks) <= {RankingSource.DATA_DRIVEN.value}
        for candidate in data_only.candidates
    )
    assert data_only.metadata["forward_top_fault_ids"] == []
    assert data_only.metadata["backward_top_fault_ids"] == []
    assert bundle.receipts["A0_DATA_ONLY"].enabled_sources == (
        RankingSource.DATA_DRIVEN.value,
    )
    assert (tmp_path / "artifacts" / "ablation" / "frozen-plan-1" / "episode-1.json").is_file()


def test_artifact_root_rejects_source_tree_and_traversal(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ArtifactPathError, match="outside"):
        ArtifactRoot(source / "outputs", source_root=source)

    root = ArtifactRoot(tmp_path / "artifacts", source_root=source)
    with pytest.raises(ArtifactPathError, match="relative"):
        root.resolve(tmp_path / "absolute.json")
    with pytest.raises(ArtifactPathError, match="traversal"):
        root.resolve("../escape.json")
    with pytest.raises(ArtifactPathError, match="name a file"):
        root.resolve("")


def test_artifact_root_rejects_symlink_escape(tmp_path: Path) -> None:
    root_path = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    root_path.mkdir()
    outside.mkdir()
    try:
        (root_path / "link").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not available on this platform")
    root = ArtifactRoot(root_path)
    with pytest.raises(ArtifactPathError, match="symlink"):
        root.write_text("link/escape.txt", "no")


def test_artifact_root_generated_writer_is_atomic_and_cleans_failure(tmp_path):
    root = ArtifactRoot(tmp_path / "artifacts")
    target = root.write_generated("models/model.bin", lambda path: path.write_bytes(b"complete"))
    assert target.read_bytes() == b"complete"

    def fail_after_partial(path):
        path.write_bytes(b"partial")
        raise RuntimeError("serializer failed")

    with pytest.raises(RuntimeError, match="serializer failed"):
        root.write_generated("models/failed.bin", fail_after_partial)
    assert not (tmp_path / "artifacts" / "models" / "failed.bin").exists()
    assert not list((tmp_path / "artifacts" / "models").glob(".failed.bin.*.tmp"))



def test_artifact_root_tolerates_directory_fsync_ebadf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression for Windows/CRT directory commit surfacing as Errno 9."""
    import astrograph.assurance.artifact_paths as artifact_paths

    real_fsync = artifact_paths.os.fsync

    def fsync_with_directory_ebadf(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        if stat.S_ISDIR(mode):
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_fsync(fd)

    monkeypatch.setattr(artifact_paths.os, "fsync", fsync_with_directory_ebadf)
    root = ArtifactRoot(tmp_path / "artifacts")
    target = root.write_json("pipeline/result.json", {"status": "ok"})
    assert target.is_file()
    assert target.read_text(encoding="utf-8").strip().endswith("}")

def test_a7_suppresses_sparse_kg_universes_but_a6_preserves_baseline() -> None:
    plan = _plan().model_copy(update={"fault_universe": ("F-A", "F-B", "F-C")})
    data = _ranking(RankingSource.DATA_DRIVEN).model_copy(
        update={
            "candidates": [
                *_ranking(RankingSource.DATA_DRIVEN).candidates,
                RankedFault(
                    fault_id="F-C",
                    rank=3,
                    score=0.1,
                    source=RankingSource.DATA_DRIVEN,
                    source_evidence_ids=["E-DATA-C"],
                    correlation_group_ids=["CORR-DATA-C"],
                    lineage_complete=True,
                ),
            ]
        }
    )
    forward = _ranking(RankingSource.KG_FORWARD)
    backward = _ranking(RankingSource.KG_BACKWARD)

    bundle = FrozenAblationRunner(plan).run(
        data_factory=lambda: data,
        forward_factory=lambda: forward,
        backward_factory=lambda: backward,
    )

    a6 = bundle.results["A6_ALL_UNADJUSTED"]
    a7 = bundle.results["A7_ALL_LINEAGE_ADJUSTED"]
    assert a6.metadata["effective_branch_weights"]["KG_FORWARD"] == 1.0
    assert a7.metadata["effective_branch_weights"]["KG_FORWARD"] == 0.0
    assert a7.metadata["effective_branch_weights"]["KG_BACKWARD"] == 0.0
    assert a7.metadata["kg_fault_universe_coverage"] == {
        "KG_FORWARD": pytest.approx(2 / 3),
        "KG_BACKWARD": pytest.approx(2 / 3),
    }
    assert a7.top_fault_id == "F-A"


def test_artifact_root_tolerates_regular_file_fsync_ebadf_when_descriptor_is_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression for Windows desktop CRT _commit returning Errno 9 on a live file."""
    import astrograph.assurance.artifact_paths as artifact_paths

    real_fsync = artifact_paths.os.fsync
    calls = {"regular": 0}

    def fsync_with_regular_ebadf(fd: int) -> None:
        mode = os.fstat(fd).st_mode
        if stat.S_ISREG(mode):
            calls["regular"] += 1
            raise OSError(errno.EBADF, "Bad file descriptor")
        real_fsync(fd)

    monkeypatch.setattr(artifact_paths.os, "fsync", fsync_with_regular_ebadf)
    root = ArtifactRoot(tmp_path / "artifacts")
    target = root.write_json("pipeline/result.json", {"status": "ok"})
    assert calls["regular"] >= 1
    assert target.is_file()
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "ok"}
