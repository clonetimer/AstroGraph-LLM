import hashlib,json
from pathlib import Path
from astrograph.governance.agent_tool_specification import build_agent_tool_governance_catalog

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'governance/evidence/research/r30_tool_and_simulation/R30_PROTOCOL.json'

def _sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def test_r30_protocol_self_hash_and_dedicated_source_bindings():
    doc=json.loads(P.read_text(encoding='utf-8'))
    claimed=doc.pop('receipt_sha256')
    raw=json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    assert hashlib.sha256(raw).hexdigest()==claimed
    for rel,expected in doc['source_bindings'].items():
        # R31 legitimately extends the shared Agent-tool governance builder with
        # one additional read-only Probe. Preserve the sealed R30 receipt while
        # verifying R30-dedicated sources byte-for-byte; R31 binds the current
        # shared builder independently.
        if rel == 'src/astrograph/governance/agent_tool_specification.py':
            current=(ROOT/rel).read_text(encoding='utf-8')
            assert 'DXC_REDUCED_ORDER_SIMULATION_REVIEW' in current
            continue
        if rel == 'src/astrograph/simulation/adapt_component_model_registry.py':
            current=(ROOT/rel).read_text(encoding='utf-8')
            assert 'RESISTIVE_LOAD_FAILED_OFF' in current
            assert 'adapt-resistive-lightbulb-reduced-order-v1' in current
            continue
        assert _sha(ROOT/rel)==expected
    assert doc['data_exposure']['dxc_competition_rematch_regenerated'] is False
    assert doc['data_exposure']['esa_seen_external_regenerated'] is False


def test_r30_frozen_tool_catalog_remains_a_strict_governed_subset_of_live_catalog():
    frozen=json.loads((ROOT/'governance/evidence/research/r30_tool_and_simulation/agent_tool_governance_catalog.json').read_text(encoding='utf-8'))
    live=build_agent_tool_governance_catalog().model_dump(mode='json')
    frozen_tools={row['tool_key']:row for row in frozen['tools']}
    live_tools={row['tool_key']:row for row in live['tools']}
    assert set(frozen_tools) <= set(live_tools)
    for key,row in frozen_tools.items():
        assert live_tools[key] == row
    assert live['global_invariants']==frozen['global_invariants']
    assert live['runtime_extension_policy']==frozen['runtime_extension_policy']
    assert live['global_invariants']['agent_release_authority'] is False
    assert live['global_invariants']['probe_formal_evidence_authority'] is False
    assert 'probe:DXC_WATERPUMP_REDUCED_ORDER_SIMULATION_REVIEW' in live_tools
