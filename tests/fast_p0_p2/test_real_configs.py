import json
from pathlib import Path
from astrograph.assurance.label_ontology_v3 import LabelOntologyBundle
from astrograph.assurance.feature_semantic_contract_v3 import FeatureSemanticContractV3
ROOT=Path(__file__).resolve().parents[2]
def load(p): return json.loads((ROOT/p).read_text(encoding="utf-8"))
def test_label_bundles_cover_full_declared_universes_without_rca_inference():
 a=LabelOntologyBundle.from_dict(load('configs/assurance/label_ontology_v3.adcs_v5_2_2.json'))
 p=LabelOntologyBundle.from_dict(load('configs/assurance/label_ontology_v3.phm_v6_0.json'))
 assert len(a.records)==14 and len(p.records)==17
 assert all(r.causal_role.value=='UNKNOWN' for r in a.records+p.records)
def test_adcs_feature_semantic_contract_remains_unfrozen_with_explicit_gaps():
 d=load('configs/assurance/feature_semantic_contract_v3.adcs_p2_innovation.DRAFT.json')
 core={k:v for k,v in d.items() if k not in {'computed_contract_sha256','unresolved_semantics'}}
 c=FeatureSemanticContractV3(**core)
 assert c.status=='AUDITED' and c.model_package_sha256 is None and c.unresolved_semantics()
def test_alignment_counts_preserve_coverage_boundaries():
 a=load('configs/assurance/ontology_alignment.adcs_v5_2_2.json')['counts']
 p=load('configs/assurance/ontology_alignment.phm_v6_0.json')['counts']
 assert (a['dataset'],a['strict_registered'],a['executable'])==(14,14,4)
 assert (p['dataset'],p['strict_registered'],p['executable'])==(17,5,5)
def test_external_manifests_have_correct_units():
 s=load('configs/external_validation/smap_msl.adapter_manifest.json'); d=load('configs/external_validation/dxc09_adapt.adapter_manifest.json'); e=load('configs/external_validation/esa_mission1.quarantine_plan.json')
 assert s['channel_count']==82 and s['independent_unit'].startswith('CHANNEL')
 assert d['experiment_group_count']>0 and d['independent_unit'].startswith('EXPERIMENT_ID')
 assert e['normal_runtime_deserialization_allowed'] is False
