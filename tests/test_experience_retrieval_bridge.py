import hashlib
from pathlib import Path

from astrograph.data_driven.experience import ExperienceLedger, ExperienceRecord
from astrograph.data_driven.memory_layers import MemoryEntry, MemoryLayer, MemoryStore
from astrograph.evidence import EpisodeEvidenceBundle
from astrograph.retrieval import (
    EvidenceAwarePreGenerationContextBuilder,
    ExperienceKnowledgeCandidateInbox,
    DiagnosticRetrievalQuery,
    GovernedTextChunk,
    OntologyConcept,
    PreReasoningRetrievalRequest,
    PreReasoningRetrievalService,
    build_experience_retrieval_assets,
)
from astrograph.orchestration.whole_spacecraft_online_workflow import WholeSpacecraftOnlineInput
from astrograph.simulation import SimulationRunCompleted


def test_only_reviewed_fault_bound_assets_enter_retrieval_and_kg_stays_unchanged(tmp_path):
    ledger = ExperienceLedger(tmp_path / 'experience')
    ledger.record(ExperienceRecord(
        experience_id='EXP-1', source_cycle_id='cycle-1', dataset_id='d',
        proposal_summary='proposal', decision='WITHHELD',
    ))
    ledger.backfill_outcome('EXP-1', outcome='confirmed', lesson='inspect heater current first')
    ledger.approve_reuse(
        'EXP-1', reviewer='human', fault_ids=['THERMAL_HEATER_STUCK_ON'],
    )
    ledger.record(ExperienceRecord(
        experience_id='EXP-UNREVIEWED', source_cycle_id='cycle-2', dataset_id='d',
        proposal_summary='proposal', decision='WITHHELD',
    ))
    memory = MemoryStore(tmp_path / 'memory')
    memory.write(MemoryEntry(
        entry_id='MEM-1', layer=MemoryLayer.SEMANTIC,
        content='temperature slope and current must agree',
        fault_ids=['THERMAL_HEATER_STUCK_ON'],
    ))
    memory.write(MemoryEntry(
        entry_id='MEM-NEG', layer=MemoryLayer.SEMANTIC, content='unsafe',
        fault_ids=['THERMAL_HEATER_STUCK_ON'], is_negative=True,
    ))

    chunks, receipt, candidates = build_experience_retrieval_assets(
        eligible_fault_ids=['THERMAL_HEATER_STUCK_ON'],
        experience_ledger_path=ledger.root,
        memory_store_path=memory.root,
    )

    assert {item.chunk_id for item in chunks} == {'experience:EXP-1', 'memory:MEM-1'}
    assert all(item.lane.value == 'EXPERIENCE' for item in chunks)
    assert receipt.active_graph_modified is False
    assert receipt.selected_experience_ids == ['EXP-1']
    assert candidates[0].review_status == 'HUMAN_REVIEW_REQUIRED'
    assert candidates[0].active_graph_modified is False
    inbox = ExperienceKnowledgeCandidateInbox(tmp_path / 'kg_curation_inbox.jsonl')
    assert inbox.stage(candidates[0]) is True
    assert inbox.stage(candidates[0]) is False
    assert inbox.records() == candidates


def test_evidence_aware_online_builder_merges_reviewed_experience_lane(tmp_path):
    ledger = ExperienceLedger(tmp_path / 'experience')
    ledger.record(ExperienceRecord(
        experience_id='EXP-ONLINE', source_cycle_id='cycle-online', dataset_id='phm-v6',
        proposal_summary='heater review', decision='HUMAN_REVIEW',
    ))
    ledger.backfill_outcome('EXP-ONLINE', outcome='confirmed', lesson='compare heater current and temperature rise')
    ledger.approve_reuse(
        'EXP-ONLINE', reviewer='human', fault_ids=['THERMAL_HEATER_STUCK_ON'],
    )
    bundle = EpisodeEvidenceBundle.model_validate_json(
        Path('tests/fixtures/evidence/heater_stuck_on/diagnosis/episode_evidence_bundle.json')
        .read_text(encoding='utf-8')
    )
    request = EvidenceAwarePreGenerationContextBuilder(
        runtime_evidence_path='configs/datasets/phm_v6_0/runtime_evidence.json',
        fault_catalog_path='configs/fault_modes/lab_fault_catalog.json',
        experience_ledger_path=ledger.root,
    ).build_request(bundle)

    assert request is not None
    assert request.experience_reuse_receipt['selected_experience_ids'] == ['EXP-ONLINE']
    result = PreReasoningRetrievalService().retrieve(request)
    assert result.experience_hit_count == 1
    assert result.experience_source_ids == ['experience:EXP-ONLINE']
    assert result.knowledge_update_candidates[0]['active_graph_modified'] is False


def test_dataset_profile_may_supply_fault_catalog_for_experience_reuse(tmp_path):
    event = SimulationRunCompleted(
        event_id='EVENT-1', run_id='RUN-1', run_dir=str(tmp_path / 'run'),
        nominal_run_dir=str(tmp_path / 'nominal'),
    )
    request = WholeSpacecraftOnlineInput(
        simulation_event=event,
        dataset_profile_id='adcs.v5_2_2',
        artifact_output_dir=str(tmp_path / 'artifacts'),
        pre_reasoning_experience_ledger_path=str(tmp_path / 'experience'),
    )
    assert request.pre_reasoning_fault_catalog_path is None


def test_only_retrieved_experience_may_leave_a_knowledge_candidate() -> None:
    def chunk(experience_id: str, text: str) -> GovernedTextChunk:
        return GovernedTextChunk(
            chunk_id=f'experience:{experience_id}',
            source_id=f'experience:{experience_id}',
            source_uri=f'memory://{experience_id}', source_sha256='a' * 64,
            chunk_sha256=hashlib.sha256(text.encode()).hexdigest(),
            knowledge_version='1', text=text, concept_ids=['fault:F-1'],
            lane='EXPERIENCE',
        )

    request = PreReasoningRetrievalRequest(
        query=DiagnosticRetrievalQuery(
            query_id='Q-1', episode_id='EP-1', text='alpha alpha',
            required_concept_ids=['fault:F-1'], seed_concept_ids=['fault:F-1'], top_k=1,
        ),
        chunks=[chunk('EXP-HIT', 'alpha alpha'), chunk('EXP-NOT-HIT', 'beta beta')],
        concepts=[OntologyConcept(
            concept_id='fault:F-1', labels=['alpha'], source_id='S',
            source_sha256='b' * 64, knowledge_version='1',
        )],
        knowledge_update_candidates=[
            {'experience_id': 'EXP-HIT', 'active_graph_modified': False},
            {'experience_id': 'EXP-NOT-HIT', 'active_graph_modified': False},
        ],
    )
    result = PreReasoningRetrievalService().retrieve(request)
    assert result.experience_source_ids == ['experience:EXP-HIT']
    assert result.knowledge_update_candidates == [
        {'experience_id': 'EXP-HIT', 'active_graph_modified': False}
    ]
