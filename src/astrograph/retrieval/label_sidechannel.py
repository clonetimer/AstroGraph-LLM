"""Development-only attacks on document identifier and layout label leakage."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from .document_corpus import CorpusSplitRole, DocumentCorpusManifest
from .document_governance import DocumentVisibilityClass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _exact_one_sided_mcnemar(
    truth: list[str], attack: list[str], baseline: list[str]
) -> tuple[int, int, float]:
    attack_only = sum(a == y and b != y for y, a, b in zip(truth, attack, baseline))
    baseline_only = sum(a != y and b == y for y, a, b in zip(truth, attack, baseline))
    discordant = attack_only + baseline_only
    if discordant == 0:
        return attack_only, baseline_only, 1.0
    probability = sum(
        math.comb(discordant, value) for value in range(attack_only, discordant + 1)
    ) / (2 ** discordant)
    return attack_only, baseline_only, min(1.0, probability)


def _identifier_text(bundle: object) -> str:
    values = [bundle.bundle_id, bundle.opaque_episode_id]
    for artifact in bundle.artifacts:
        if artifact.descriptor.visibility_class == DocumentVisibilityClass.OBSERVATION_CAUSAL:
            values.extend([
                artifact.relative_path,
                artifact.descriptor.document_id,
                artifact.descriptor.source_uri,
            ])
    return " ".join(values)


def _layout_features(bundle: object, *, corpus_root: Path) -> list[float]:
    rows: list[list[float]] = []
    for artifact in bundle.artifacts:
        if artifact.descriptor.visibility_class != DocumentVisibilityClass.OBSERVATION_CAUSAL:
            continue
        path = (corpus_root / artifact.relative_path).resolve()
        if not path.is_relative_to(corpus_root) or not path.is_file():
            raise ValueError("runtime document path is missing or escapes corpus root")
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        rows.append([
            float(path.stat().st_size),
            float(len(lines)),
            float(sum(line.lstrip().startswith("#") for line in lines)),
            float(sum(line.lstrip().startswith("|") for line in lines)),
        ])
    if not rows:
        raise ValueError("side-channel attack bundle has no causal runtime documents")
    values = np.asarray(rows, dtype=float)
    return [
        float(len(rows)),
        *values.sum(axis=0).tolist(),
        *values.mean(axis=0).tolist(),
        *values.min(axis=0).tolist(),
        *values.max(axis=0).tolist(),
    ]


def _metrics(
    truth: list[str], predictions: list[str], baseline: list[str]
) -> dict[str, Any]:
    attack_only, baseline_only, p_value = _exact_one_sided_mcnemar(
        truth, predictions, baseline
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro", zero_division=0)),
        "attack_correct_baseline_wrong": attack_only,
        "attack_wrong_baseline_correct": baseline_only,
        "paired_exact_mcnemar_one_sided_p": p_value,
        "significantly_above_majority_at_alpha_0_05": p_value < 0.05,
    }


def run_document_label_sidechannel_attack(
    *,
    preregistration_path: str | Path,
    manifest_path: str | Path,
    corpus_root: str | Path,
    dataset_index_paths: dict[str, str | Path],
) -> dict[str, Any]:
    preregistration_path = Path(preregistration_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    corpus_root = Path(corpus_root).resolve()
    prereg = json.loads(preregistration_path.read_text(encoding="utf-8"))
    if not prereg.get("frozen_before_execution"):
        raise ValueError("side-channel attack preregistration is not frozen")
    manifest = DocumentCorpusManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )
    seed = int(prereg["random_seed"])
    input_hashes = {
        "preregistration": _sha256(preregistration_path),
        "manifest": _sha256(manifest_path),
    }
    row_maps: dict[str, dict[str, dict[str, str]]] = {}
    for dataset_id, raw_path in sorted(dataset_index_paths.items()):
        path = Path(raw_path).resolve()
        input_hashes[f"index:{dataset_id}"] = _sha256(path)
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        row_maps[dataset_id] = {
            hashlib.sha256(row["sample_id"].encode()).hexdigest(): row for row in rows
        }

    forbidden = set(int(item) for item in prereg["forbidden_groups"])
    results: dict[str, Any] = {}
    violations: list[str] = []
    for dataset_id in sorted(dataset_index_paths):
        records: list[tuple[object, dict[str, str]]] = []
        for bundle in manifest.bundles:
            if bundle.dataset_id != dataset_id:
                continue
            row = row_maps[dataset_id].get(bundle.source_episode_ref_sha256)
            if row is None:
                raise ValueError("manifest bundle cannot be mapped to its source index")
            group = int(row["sample_index"])
            if group in forbidden:
                raise ValueError("forbidden post-hoc group entered attack materialization")
            records.append((bundle, row))
        train = [item for item in records if item[0].split_role == CorpusSplitRole.TRAIN]
        evaluation = [
            item for item in records if item[0].split_role == CorpusSplitRole.DEVELOPMENT
        ]
        if not train or not evaluation:
            raise ValueError("attack requires nonempty train and development records")
        train_labels = [row["diagnostic_class"] for _, row in train]
        eval_labels = [row["diagnostic_class"] for _, row in evaluation]
        majority = sorted(Counter(train_labels).items(), key=lambda item: (-item[1], item[0]))[0][0]
        baseline = [majority] * len(eval_labels)

        train_identifiers = [_identifier_text(bundle) for bundle, _ in train]
        eval_identifiers = [_identifier_text(bundle) for bundle, _ in evaluation]
        vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(2, 5), min_df=2, max_features=50_000,
            lowercase=True, sublinear_tf=True,
        )
        train_identifier_matrix = vectorizer.fit_transform(train_identifiers)
        eval_identifier_matrix = vectorizer.transform(eval_identifiers)
        train_layout = np.asarray([
            _layout_features(bundle, corpus_root=corpus_root) for bundle, _ in train
        ])
        eval_layout = np.asarray([
            _layout_features(bundle, corpus_root=corpus_root) for bundle, _ in evaluation
        ])
        scaler = StandardScaler()
        train_layout_matrix = csr_matrix(scaler.fit_transform(train_layout))
        eval_layout_matrix = csr_matrix(scaler.transform(eval_layout))

        lane_matrices = {
            "IDENTIFIER_ONLY": (train_identifier_matrix, eval_identifier_matrix),
            "LAYOUT_ONLY": (train_layout_matrix, eval_layout_matrix),
            "IDENTIFIER_PLUS_LAYOUT": (
                hstack([train_identifier_matrix, train_layout_matrix], format="csr"),
                hstack([eval_identifier_matrix, eval_layout_matrix], format="csr"),
            ),
        }
        lanes: dict[str, Any] = {}
        for lane, (train_matrix, eval_matrix) in lane_matrices.items():
            classifier = LogisticRegression(
                C=1.0, max_iter=2_000, random_state=seed, solver="lbfgs"
            )
            classifier.fit(train_matrix, train_labels)
            predictions = classifier.predict(eval_matrix).tolist()
            lane_result = _metrics(eval_labels, predictions, baseline)
            lanes[lane] = lane_result
            if lane_result["significantly_above_majority_at_alpha_0_05"]:
                violations.append(f"{dataset_id}:{lane}:SIGNIFICANT_LABEL_SIDECHANNEL")
        results[dataset_id] = {
            "train_count": len(train),
            "evaluation_count": len(evaluation),
            "class_count": len(set(train_labels) | set(eval_labels)),
            "majority_class": majority,
            "majority_accuracy": float(accuracy_score(eval_labels, baseline)),
            "lanes": lanes,
        }

    body = {
        "schema_version": "astrograph-document-label-sidechannel-report.v1",
        "experiment_id": prereg["experiment_id"],
        "status": "PASS" if not violations else "FAIL",
        "input_sha256": dict(sorted(input_hashes.items())),
        "library_versions": {
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        "datasets": results,
        "violations": violations,
        "posthoc_telemetry_labels_or_truth_opened": False,
        "document_body_tokens_used": False,
        "claim_boundary": prereg["claim_boundary"],
    }
    return {**body, "report_sha256": _canonical_sha(body)}


__all__ = ["run_document_label_sidechannel_attack"]
