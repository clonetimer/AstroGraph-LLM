from .release_policy import D3_KNOWLEDGE_VERSION, D3_RELEASE_ID, build_d3_rw_candidate_release
from .governance import (
    JsonKnowledgeGovernanceStore,
    KnowledgeChangeRequest,
    KnowledgeChangeStatus,
    KnowledgeChangeTransition,
)
from .release import KnowledgeRelease, sha256_file, validate_knowledge_release
from .repository import CsvKnowledgeRepository, KnowledgeRepository, Neo4jKnowledgeRepository, csv_repository_from_release
from .soft_projection import (
    SoftGraphFaultAssessment,
    SoftGraphInferenceResult,
    SoftGraphObservation,
    SoftGraphIndependenceDeclaration,
    SoftGraphProvenancePolicy,
    SoftGraphSuggestedAction,
    build_soft_graph_inference,
    load_soft_graph_provenance_policy,
)
from .patches import (
    AstroGraphKnowledgePatchReplayContract,
    AstroGraphKnowledgePatchReplayEpisode,
    AstroGraphKnowledgePatchReplayOutput,
    CompiledKnowledgePatch,
    KnowledgeEdgeValue,
    KnowledgePatchCandidate,
    KnowledgePatchImpactReport,
    KnowledgePatchOperation,
    KnowledgePatchOperationKind,
    KnowledgePatchSourceSplit,
    KnowledgePatchValidationReport,
    build_astrograph_knowledge_patch_replay_output,
    build_knowledge_patch_candidate,
    compile_knowledge_patch_release,
    experience_candidate_is_compilable,
    knowledge_release_sha256,
    run_knowledge_patch_impact_replay,
    run_astrograph_knowledge_patch_impact_replay,
    validate_knowledge_patch,
)
from .coverage_review import (
    DatasetKGReviewPackageManifest,
    FaultCoverageReviewRow,
    KGReviewSourceAnchor,
    KnowledgeMappingCandidateSet,
    KnowledgeMappingProposal,
    ReplayEpisodeCandidate,
    ReplayScenarioSetCandidate,
    SourceAdjudicationPackage,
    SourceAdjudicationRecord,
    SupplementalReviewFreezeReceipt,
    SupplementalReviewRefreezeReceipt,
    SupplementalReviewUnfreezeReceipt,
    build_source_adjudication_package,
    build_dataset_kg_review_package,
    freeze_supplemental_human_review,
    refreeze_supplemental_human_review,
    unfreeze_supplemental_human_review,
    validate_dataset_kg_review_package,
    validate_source_adjudication_package,
    validate_supplemental_review_freeze,
    validate_supplemental_review_refreeze,
    validate_supplemental_review_unfreeze,
    write_source_adjudication_review_sheet,
)
from .tbox_approval import (
    TBoxVocabularyApproval,
    approve_cross_domain_tbox_vocabulary,
    write_tbox_vocabulary_approval,
)

__all__ = [
    "TBoxVocabularyApproval",
    "approve_cross_domain_tbox_vocabulary",
    "write_tbox_vocabulary_approval",
    "D3_KNOWLEDGE_VERSION",
    "D3_RELEASE_ID",
    "build_d3_rw_candidate_release",
    "KnowledgeRelease",
    "sha256_file",
    "validate_knowledge_release",
    "CsvKnowledgeRepository",
    "KnowledgeRepository",
    "Neo4jKnowledgeRepository",
    "csv_repository_from_release",
    "JsonKnowledgeGovernanceStore",
    "KnowledgeChangeRequest",
    "KnowledgeChangeStatus",
    "KnowledgeChangeTransition",
    "SoftGraphFaultAssessment",
    "SoftGraphInferenceResult",
    "SoftGraphObservation",
    "SoftGraphIndependenceDeclaration",
    "SoftGraphProvenancePolicy",
    "SoftGraphSuggestedAction",
    "build_soft_graph_inference",
    "load_soft_graph_provenance_policy",
    "CompiledKnowledgePatch",
    "AstroGraphKnowledgePatchReplayContract",
    "AstroGraphKnowledgePatchReplayEpisode",
    "AstroGraphKnowledgePatchReplayOutput",
    "KnowledgeEdgeValue",
    "KnowledgePatchCandidate",
    "KnowledgePatchImpactReport",
    "KnowledgePatchOperation",
    "KnowledgePatchOperationKind",
    "KnowledgePatchSourceSplit",
    "KnowledgePatchValidationReport",
    "build_astrograph_knowledge_patch_replay_output",
    "build_knowledge_patch_candidate",
    "compile_knowledge_patch_release",
    "experience_candidate_is_compilable",
    "knowledge_release_sha256",
    "run_knowledge_patch_impact_replay",
    "run_astrograph_knowledge_patch_impact_replay",
    "validate_knowledge_patch",
    "DatasetKGReviewPackageManifest",
    "FaultCoverageReviewRow",
    "KGReviewSourceAnchor",
    "KnowledgeMappingCandidateSet",
    "KnowledgeMappingProposal",
    "ReplayEpisodeCandidate",
    "ReplayScenarioSetCandidate",
    "SourceAdjudicationPackage",
    "SourceAdjudicationRecord",
    "SupplementalReviewFreezeReceipt",
    "SupplementalReviewRefreezeReceipt",
    "SupplementalReviewUnfreezeReceipt",
    "build_source_adjudication_package",
    "build_dataset_kg_review_package",
    "freeze_supplemental_human_review",
    "refreeze_supplemental_human_review",
    "unfreeze_supplemental_human_review",
    "validate_dataset_kg_review_package",
    "validate_source_adjudication_package",
    "validate_supplemental_review_freeze",
    "validate_supplemental_review_refreeze",
    "validate_supplemental_review_unfreeze",
    "write_source_adjudication_review_sheet",
]
from .evidence_graph import (
    EpisodeEvidenceGraph,
    EvidenceGraphEdge,
    EvidenceGraphNode,
    build_episode_evidence_graph,
)

__all__ += [
    "EpisodeEvidenceGraph",
    "EvidenceGraphEdge",
    "EvidenceGraphNode",
    "build_episode_evidence_graph",
]

from .query import (
    CompiledGraphQuery,
    CsvEvidenceGraphReferenceExecutor,
    GraphEvidenceAssociation,
    GraphQueryExecutionResult,
    GraphQueryKind,
    GraphQueryRow,
    ReadOnlyCypherCompiler,
    ReadOnlyGraphQueryExecutor,
    ReadOnlyGraphQueryPolicy,
    associations_from_query_result,
    build_backward_fault_query_plan,
    build_forward_evidence_query_plan,
    validate_read_only_cypher,
)

__all__ += [
    "CompiledGraphQuery",
    "CsvEvidenceGraphReferenceExecutor",
    "GraphEvidenceAssociation",
    "GraphQueryExecutionResult",
    "GraphQueryKind",
    "GraphQueryRow",
    "ReadOnlyCypherCompiler",
    "ReadOnlyGraphQueryExecutor",
    "ReadOnlyGraphQueryPolicy",
    "associations_from_query_result",
    "build_backward_fault_query_plan",
    "build_forward_evidence_query_plan",
    "validate_read_only_cypher",
]

from .schema import (
    AssertionStatus,
    DEFAULT_KNOWLEDGE_SCHEMA,
    KnowledgeNodeType,
    KnowledgeRelationType,
    KnowledgeSchema,
    ReviewStatus,
)

from .schema_v3 import (
    DEFAULT_GRAPH_SHAPE_VALIDATOR,
    GraphEdgeV3,
    GraphLayer,
    GraphNodeV3,
    GraphShapeIssue,
    GraphShapeReport,
    GraphShapeValidator,
    KnowledgeGraphV3,
    KnowledgeNodeTypeV3,
    KnowledgeRelationTypeV3,
    LifecycleStatus,
    NODE_LAYERS,
    RELATION_ALLOWLIST,
)
from .abstract_fault_ontology import (
    DatasetFaultBinding,
    DatasetFaultOntologyBindings,
    GenericEffectPattern,
    GenericFailureMechanism,
    GenericFaultOntology,
    RobustnessContract,
    build_generic_fault_ontology_bundle,
    load_generic_fault_ontology,
    load_dataset_fault_bindings,
    project_generic_fault_ontology,
)
from .spatiotemporal_fault_graph import (
    ComponentEpisodeEdge,
    ComponentEpisodeNode,
    ComponentSpatiotemporalFaultGraph,
    ComponentTopologyEdge,
    EpisodeComponentBinding,
    SpatiotemporalFaultTemplate,
    SpatiotemporalFaultTemplateCatalog,
    SpatiotemporalRelationType,
    SpatiotemporalTemplateMatch,
    TemplateEdgeRequirement,
    TemplateNodeRequirement,
    build_component_spatiotemporal_graph,
    match_spatiotemporal_templates,
)
from .causal_paths import (
    CausalEvidencePath,
    CausalMetaPathQueryService,
    CausalMetaPathTemplate,
    CausalPathExclusion,
    CausalPathHop,
    CausalPathQueryPolicy,
    CausalPathQueryResult,
)
from .review_readiness import (
    CausalEdgeReviewItem,
    CausalEdgeReviewPacket,
    ReviewPacketValidationReceipt,
    ReviewSourceAnchor,
    StructuredCandidateBenchmarkInput,
    StructuredEntityLinkingBenchmark,
    StructuredEntityLinkingBenchmarkCase,
    load_frozen_structured_entity_linking_benchmark,
    validate_causal_edge_review_packet,
)

__all__ += [
    "DEFAULT_GRAPH_SHAPE_VALIDATOR",
    "GraphEdgeV3",
    "GraphLayer",
    "GraphNodeV3",
    "GraphShapeIssue",
    "GraphShapeReport",
    "GraphShapeValidator",
    "KnowledgeGraphV3",
    "KnowledgeNodeTypeV3",
    "KnowledgeRelationTypeV3",
    "LifecycleStatus",
    "NODE_LAYERS",
    "RELATION_ALLOWLIST",
    "CanonicalConcept",
    "CanonicalRelation",
    "CanonicalizationAction",
    "CanonicalizationDecision",
    "KnowledgeCanonicalizationReport",
    "KnowledgeConstructionBundle",
    "KnowledgeConstructionQualityReport",
    "DocumentKnowledgeGraphScope",
    "DocumentKnowledgeStagingBundle",
    "DocumentProcessingCrosswalk",
    "GovernedDocumentProcessingBundle",
    "KnowledgeExpansionSuggestion",
    "KnowledgeExtractionPassKind",
    "KnowledgeExtractionTask",
    "KnowledgeSourceRole",
    "SourceSection",
    "SourceStructureTree",
    "assess_candidate_graph",
    "build_extraction_tasks",
    "build_knowledge_construction_bundle",
    "canonicalize_knowledge_proposal",
    "compile_candidate_graph",
    "knowledge_source_profile_set_sha256",
    "govern_document_knowledge_proposal",
    "process_governed_document",
    "process_governed_docx",
    "profile_structured_source",
    "stage_governed_document_source",
    "stage_quarantined_document_source",
    "suggest_hidden_edges",
    "write_construction_bundle",
    "GenericEffectPattern",
    "DatasetFaultBinding",
    "DatasetFaultOntologyBindings",
    "GenericFailureMechanism",
    "GenericFaultOntology",
    "RobustnessContract",
    "build_generic_fault_ontology_bundle",
    "load_generic_fault_ontology",
    "load_dataset_fault_bindings",
    "project_generic_fault_ontology",
    "ComponentEpisodeEdge",
    "ComponentEpisodeNode",
    "ComponentSpatiotemporalFaultGraph",
    "ComponentTopologyEdge",
    "EpisodeComponentBinding",
    "SpatiotemporalFaultTemplate",
    "SpatiotemporalFaultTemplateCatalog",
    "SpatiotemporalRelationType",
    "SpatiotemporalTemplateMatch",
    "TemplateEdgeRequirement",
    "TemplateNodeRequirement",
    "build_component_spatiotemporal_graph",
    "match_spatiotemporal_templates",
    "CausalEvidencePath",
    "CausalMetaPathQueryService",
    "CausalMetaPathTemplate",
    "CausalPathExclusion",
    "CausalPathHop",
    "CausalPathQueryPolicy",
    "CausalPathQueryResult",
    "CausalEdgeReviewItem",
    "CausalEdgeReviewPacket",
    "ReviewPacketValidationReceipt",
    "ReviewSourceAnchor",
    "StructuredCandidateBenchmarkInput",
    "StructuredEntityLinkingBenchmark",
    "StructuredEntityLinkingBenchmarkCase",
    "load_frozen_structured_entity_linking_benchmark",
    "validate_causal_edge_review_packet",
]

from .runtime import (
    KGEvidenceRuntimeReference,
    KGEvidenceRuntimeResult,
    KGEvidenceRuntimeService,
    Neo4jKGEvidenceBindingConfig,
)

__all__ += [
    "KGEvidenceRuntimeReference",
    "KGEvidenceRuntimeResult",
    "KGEvidenceRuntimeService",
    "Neo4jKGEvidenceBindingConfig",
]

__all__ += [
    "audit_document_identity_relocation",
    "build_document_graph_gold_handoff",
    "evaluate_document_graph_gold",
    "freeze_document_graph_review_decisions",
    "build_document_graph_patch_proposal",
    "compile_document_graph_patch_candidate",
    "replay_document_graph_patch_topology",
    "build_reviewed_tbox_retrieval_projection",
]


_CONSTRUCTION_EXPORTS = {
    "CanonicalConcept",
    "CanonicalRelation",
    "CanonicalizationAction",
    "CanonicalizationDecision",
    "KnowledgeCanonicalizationReport",
    "KnowledgeConstructionBundle",
    "KnowledgeConstructionQualityReport",
    "DocumentKnowledgeGraphScope",
    "DocumentKnowledgeStagingBundle",
    "DocumentProcessingCrosswalk",
    "GovernedDocumentProcessingBundle",
    "KnowledgeExpansionSuggestion",
    "KnowledgeExtractionPassKind",
    "KnowledgeExtractionTask",
    "KnowledgeSourceRole",
    "SourceSection",
    "SourceStructureTree",
    "assess_candidate_graph",
    "build_extraction_tasks",
    "build_knowledge_construction_bundle",
    "canonicalize_knowledge_proposal",
    "compile_candidate_graph",
    "knowledge_source_profile_set_sha256",
    "govern_document_knowledge_proposal",
    "process_governed_document",
    "process_governed_docx",
    "profile_structured_source",
    "stage_governed_document_source",
    "stage_quarantined_document_source",
    "suggest_hidden_edges",
    "write_construction_bundle",
}


def __getattr__(name: str):
    if name in _CONSTRUCTION_EXPORTS:
        from . import construction

        return getattr(construction, name)
    if name == "audit_document_identity_relocation":
        from .document_identity import audit_document_identity_relocation

        return audit_document_identity_relocation
    if name == "freeze_document_graph_review_decisions":
        from .document_graph_review_decisions import freeze_document_graph_review_decisions

        return freeze_document_graph_review_decisions
    if name == "build_document_graph_gold_handoff":
        from .document_graph_gold import build_document_graph_gold_handoff

        return build_document_graph_gold_handoff
    if name == "evaluate_document_graph_gold":
        from .document_graph_gold_evaluation import evaluate_document_graph_gold

        return evaluate_document_graph_gold
    if name in {
        "build_document_graph_patch_proposal",
        "compile_document_graph_patch_candidate",
        "replay_document_graph_patch_topology",
    }:
        from . import document_graph_patch

        return getattr(document_graph_patch, name)
    if name == "build_reviewed_tbox_retrieval_projection":
        from .tbox_retrieval_projection import build_reviewed_tbox_retrieval_projection

        return build_reviewed_tbox_retrieval_projection
    raise AttributeError(name)

