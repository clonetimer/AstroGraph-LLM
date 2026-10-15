from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import AssertionStatus, ReviewStatus

from .schema_v3 import (
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
)


class RobustnessContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_fields_belong_to: str
    units_belong_to: str
    numeric_limits_belong_to: str
    evidence_values_belong_to: str
    supported_feature_families: list[str] = Field(min_length=1)
    missing_or_out_of_distribution_policy: str


class GenericFailureMechanism(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mechanism_id: str
    name: str
    description: str
    applicable_archetype_ids: list[str] = Field(min_length=1)
    effect_pattern_ids: list[str] = Field(min_length=1)


class GenericEffectPattern(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effect_pattern_id: str
    name: str
    quantity_kind_ids: list[str] = Field(min_length=1)
    feature_family_ids: list[str] = Field(min_length=1)


class GenericFaultOntology(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-generic-fault-ontology.v1"
    ontology_id: str
    version: str
    component_catalog_id: str
    semantic_schema_id: str
    robustness_contract: RobustnessContract
    failure_mechanisms: list[GenericFailureMechanism] = Field(min_length=1)
    effect_patterns: list[GenericEffectPattern] = Field(min_length=1)
    claim_boundary: str

    @model_validator(mode="after")
    def validate_identity_and_references(self) -> "GenericFaultOntology":
        mechanism_ids = [item.mechanism_id for item in self.failure_mechanisms]
        effect_ids = [item.effect_pattern_id for item in self.effect_patterns]
        if len(mechanism_ids) != len(set(mechanism_ids)):
            raise ValueError("mechanism_id values must be unique")
        if len(effect_ids) != len(set(effect_ids)):
            raise ValueError("effect_pattern_id values must be unique")
        known_effects = set(effect_ids)
        referenced = {
            effect for mechanism in self.failure_mechanisms for effect in mechanism.effect_pattern_ids
        }
        if referenced - known_effects:
            raise ValueError(f"unknown effect patterns: {sorted(referenced - known_effects)}")
        return self


class DatasetFaultBinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fault_class_id: str
    mechanism_id: str
    expected_effect_pattern_ids: list[str] = Field(min_length=1)


class DatasetFaultOntologyBindings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "astrograph-dataset-fault-ontology-bindings.v1"
    binding_id: str
    dataset_id: str
    ontology_id: str
    ontology_version: str
    bindings: list[DatasetFaultBinding] = Field(min_length=1)
    claim_boundary: str

    @model_validator(mode="after")
    def unique_fault_classes(self) -> "DatasetFaultOntologyBindings":
        ids = [item.fault_class_id for item in self.bindings]
        if len(ids) != len(set(ids)):
            raise ValueError("fault_class_id values must be unique")
        return self


def load_generic_fault_ontology(path: str | Path) -> GenericFaultOntology:
    return GenericFaultOntology.model_validate_json(Path(path).read_text(encoding="utf-8"))


def load_dataset_fault_bindings(
    path: str | Path, ontology: GenericFaultOntology
) -> DatasetFaultOntologyBindings:
    package = DatasetFaultOntologyBindings.model_validate_json(Path(path).read_text(encoding="utf-8"))
    if (package.ontology_id, package.ontology_version) != (ontology.ontology_id, ontology.version):
        raise ValueError("dataset binding targets a different ontology version")
    mechanisms = {item.mechanism_id: item for item in ontology.failure_mechanisms}
    effects = {item.effect_pattern_id for item in ontology.effect_patterns}
    for binding in package.bindings:
        mechanism = mechanisms.get(binding.mechanism_id)
        if mechanism is None:
            raise ValueError(f"unknown bound mechanism: {binding.mechanism_id}")
        selected = set(binding.expected_effect_pattern_ids)
        if selected - effects:
            raise ValueError(f"unknown bound effects: {sorted(selected - effects)}")
        if selected - set(mechanism.effect_pattern_ids):
            raise ValueError(
                f"effects are not declared for {binding.mechanism_id}: "
                f"{sorted(selected - set(mechanism.effect_pattern_ids))}"
            )
    return package


def project_generic_fault_ontology(
    ontology_path: str | Path,
    component_catalog_path: str | Path,
    semantic_schema_path: str | Path,
) -> KnowledgeGraphV3:
    """Project the stable TBox through the reusable governed build pipeline."""
    not_applicable_refreeze = hashlib.sha256(
        b"LIBRARY_PROJECTION_WITHOUT_SUPPLEMENTAL_REVIEW_INPUT"
    ).hexdigest()
    bundle = build_generic_fault_ontology_bundle(
        ontology_path,
        component_catalog_path,
        semantic_schema_path,
        excluded_review_refreeze_sha256=not_applicable_refreeze,
    )
    if bundle.candidate_graph is None:
        raise ValueError(
            "generic fault ontology failed reusable construction: "
            f"{bundle.candidate_validation.findings}; {bundle.quality.blocker_codes}"
        )
    return bundle.candidate_graph


def build_generic_fault_ontology_bundle(
    ontology_path: str | Path,
    component_catalog_path: str | Path,
    semantic_schema_path: str | Path,
    *,
    excluded_review_refreeze_sha256: str,
) -> KnowledgeConstructionBundle:
    """Rebuild the governed cross-domain TBox as a review-bounded bundle.

    This is a migration of already governed AstroGraph assets, not a claim that
    one source is enough to establish new causal knowledge. Therefore the
    migration gate uses one independence group and cannot publish an Active KG.
    """
    from astrograph.agents.knowledge_curation import (
        KnowledgeExtractionProposal,
        KnowledgeNodeCandidate,
        KnowledgeRelationCandidate,
    )
    from .construction import (
        KnowledgeSourceRole,
        build_knowledge_construction_bundle,
        knowledge_source_profile_set_sha256,
        profile_structured_source,
    )

    ontology_path = Path(ontology_path)
    ontology = load_generic_fault_ontology(ontology_path)
    archetype_catalog = json.loads(Path(component_catalog_path).read_text(encoding="utf-8"))
    semantic_schema = json.loads(Path(semantic_schema_path).read_text(encoding="utf-8"))
    if archetype_catalog["catalog_id"] != ontology.component_catalog_id:
        raise ValueError("component catalog identity does not match ontology contract")
    if semantic_schema["semantic_schema_id"] != ontology.semantic_schema_id:
        raise ValueError("semantic schema identity does not match ontology contract")

    archetypes = {item["archetype_id"]: item for item in archetype_catalog["archetypes"]}
    quantities = {item["quantity_kind_id"]: item for item in semantic_schema["quantity_kinds"]}
    required_archetypes = {
        value for mechanism in ontology.failure_mechanisms
        for value in mechanism.applicable_archetype_ids
    }
    required_quantities = {
        value for effect in ontology.effect_patterns for value in effect.quantity_kind_ids
    }
    if required_archetypes - set(archetypes):
        raise ValueError(f"unknown component archetypes: {sorted(required_archetypes - set(archetypes))}")
    if required_quantities - set(quantities):
        raise ValueError(f"unknown quantity kinds: {sorted(required_quantities - set(quantities))}")

    source_specs = [
        (
            "source:astrograph:generic-fault-ontology",
            ontology_path,
            "astrograph-governed-cross-domain-ontology",
        ),
        (
            "source:astrograph:component-archetype-catalog",
            Path(component_catalog_path),
            "astrograph-governed-component-semantics",
        ),
        (
            "source:astrograph:physical-quantity-schema",
            Path(semantic_schema_path),
            "astrograph-governed-quantity-semantics",
        ),
    ]
    profiled = [
        profile_structured_source(
            source_id=source_id,
            source_path=path,
            independence_group=group,
            license_id="AstroGraph-project-governed-asset",
            source_role=KnowledgeSourceRole.ASTROGRAPH_ASSET,
        )
        for source_id, path, group in source_specs
    ]
    profiles = [item[0] for item in profiled]
    trees = [item[1] for item in profiled]

    def fragment(tree_index: int, title: str) -> str:
        matches = [item.fragment_id for item in trees[tree_index].sections if item.title == title]
        if len(matches) != 1:
            raise ValueError(f"expected one structured section for {title!r}; got {len(matches)}")
        return matches[0]

    nodes: list[KnowledgeNodeCandidate] = []
    candidate_ids: dict[str, str] = {}

    def add_node(
        canonical_id: str,
        node_type: KnowledgeNodeTypeV3,
        canonical_name: str,
        description: str,
        source_fragment_id: str,
        properties: dict,
    ) -> None:
        candidate_id = f"candidate:migration:{canonical_id}"
        candidate_ids[canonical_id] = candidate_id
        nodes.append(KnowledgeNodeCandidate(
            node_id=candidate_id,
            node_type=node_type,
            canonical_name=canonical_name,
            description=description,
            canonical_id_hint=canonical_id,
            properties=properties,
            source_fragment_ids=[source_fragment_id],
            confidence=1.0,
        ))

    for archetype_id in sorted(required_archetypes):
        value = archetypes[archetype_id]
        add_node(
            f"archetype:{archetype_id}",
            KnowledgeNodeTypeV3.COMPONENT_ARCHETYPE,
            archetype_id,
            value["description"],
            fragment(1, archetype_id),
            {
                "canonical_id": archetype_id,
                "family": value["family"],
                "functional_role_ids": value["functional_role_ids"],
                "description": value["description"],
            },
        )
    for mechanism in ontology.failure_mechanisms:
        add_node(
            mechanism.mechanism_id,
            KnowledgeNodeTypeV3.FAILURE_MECHANISM,
            mechanism.name,
            mechanism.description,
            fragment(0, mechanism.mechanism_id),
            {"name": mechanism.name, "description": mechanism.description},
        )
    for effect in ontology.effect_patterns:
        add_node(
            effect.effect_pattern_id,
            KnowledgeNodeTypeV3.EFFECT_PATTERN,
            effect.name,
            effect.name,
            fragment(0, effect.effect_pattern_id),
            {"name": effect.name, "feature_family_ids": effect.feature_family_ids},
        )
    for quantity_id in sorted(required_quantities):
        value = quantities[quantity_id]
        add_node(
            f"quantity:{quantity_id}",
            KnowledgeNodeTypeV3.QUANTITY_KIND,
            quantity_id,
            value["description"],
            fragment(2, quantity_id),
            {
                "canonical_id": quantity_id,
                "canonical_unit": value["canonical_unit"],
                "dimension": value["dimension"],
                "value_semantics": value["value_semantics"],
            },
        )

    relations: list[KnowledgeRelationCandidate] = []

    def add_relation(
        source: str,
        relation: KnowledgeRelationTypeV3,
        target: str,
        source_fragment_id: str,
    ) -> None:
        local = f"{relation.value.casefold()}:{source.replace(':', '.')}-{target.replace(':', '.')}"
        relations.append(KnowledgeRelationCandidate(
            edge_id=f"candidate-edge:migration:{local}",
            source_id=candidate_ids[source],
            relation_type=relation,
            target_id=candidate_ids[target],
            source_fragment_ids=[source_fragment_id],
            confidence=1.0,
            rationale=(
                "Migrated from the governed cross-domain ontology; this association is a TBox "
                "scope declaration, not episode evidence or a newly established causal claim."
            ),
            predicate_definition=f"Governed {relation.value} TBox association.",
            properties={"scope": "cross-domain TBox association; not episode evidence or causal effect"},
        ))

    for mechanism in ontology.failure_mechanisms:
        for archetype_id in mechanism.applicable_archetype_ids:
            add_relation(
                f"archetype:{archetype_id}",
                KnowledgeRelationTypeV3.SUSCEPTIBLE_TO,
                mechanism.mechanism_id,
                fragment(0, mechanism.mechanism_id),
            )
        for effect_id in mechanism.effect_pattern_ids:
            add_relation(
                mechanism.mechanism_id,
                KnowledgeRelationTypeV3.MANIFESTS_AS,
                effect_id,
                fragment(0, mechanism.mechanism_id),
            )
    for effect in ontology.effect_patterns:
        for quantity_id in effect.quantity_kind_ids:
            add_relation(
                effect.effect_pattern_id,
                KnowledgeRelationTypeV3.INDICATED_BY,
                f"quantity:{quantity_id}",
                fragment(0, effect.effect_pattern_id),
            )

    proposal = KnowledgeExtractionProposal(
        source_profile_sha256=knowledge_source_profile_set_sha256(profiles),
        nodes=nodes,
        relations=relations,
        proposer_id="astrograph-governed-ontology-migrator",
        proposer_version="1.0.0",
    )
    return build_knowledge_construction_bundle(
        source_profiles=profiles,
        source_trees=trees,
        proposal=proposal,
        approved_existing_nodes={},
        approved_extraction_methods=[item.extraction_method for item in profiles],
        excluded_review_refreeze_sha256=excluded_review_refreeze_sha256,
        graph_id=f"graph:{ontology.ontology_id}:{ontology.version}",
        build_mode="MIGRATE_EXISTING_GOVERNED_ONTOLOGY",
        minimum_independent_causal_sources=1,
        lifecycle_status=LifecycleStatus.ACTIVE,
        review_status=ReviewStatus.MACHINE_VALIDATED,
        assertion_status=AssertionStatus.DERIVED,
    )


__all__ = [
    "GenericEffectPattern",
    "GenericFailureMechanism",
    "GenericFaultOntology",
    "DatasetFaultBinding",
    "DatasetFaultOntologyBindings",
    "RobustnessContract",
    "load_generic_fault_ontology",
    "load_dataset_fault_bindings",
    "build_generic_fault_ontology_bundle",
    "project_generic_fault_ontology",
]
