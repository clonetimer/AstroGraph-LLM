"""Small deterministic two-fault fixtures, not empirical accuracy evidence."""
import csv
import json
import pytest
from astrograph.evidence.executable import (
    EpisodeWindow, FeatureDefinition, FeatureResult, EvidenceDefinition,
    EpisodeEvidenceBundle, evaluate_evidence,
)
from astrograph.fusion import FaultRanking, RankedFault, RankingSource
from astrograph.oag.contracts import OAGInput


@pytest.fixture
def make_diagnostic_case(tmp_path):
    def build(a=0.0, b=None, counter=None, data_ids=("A", "B")):
        ep=EpisodeWindow(episode_id="episode",run_id="run",paired_nominal_run_id="nominal",
                         start_s=0,end_s=2,source="numeric-test-fixture",operating_condition={"mode":"tracking"})
        defs=[FeatureDefinition(feature_definition_id=f"F_{name}",name=name,source_field_template=f"sensor.{name}",
                                aggregation="MAX_VALUE",unit="1",algorithm_version="1") for name in ("a","b","counter")]
        edefs=[EvidenceDefinition(evidence_definition_id="E_A",target_fault_id="A",name="A required",feature_definition_id="F_a",
                                 role="REQUIRED",comparison="GREATER_OR_EQUAL",threshold=1,threshold_unit="1",critical=True),
               EvidenceDefinition(evidence_definition_id="E_B",target_fault_id="B",name="B required",feature_definition_id="F_b",
                                 role="REQUIRED",comparison="GREATER_OR_EQUAL",threshold=1,threshold_unit="1",critical=True),
               EvidenceDefinition(evidence_definition_id="E_CA",target_fault_id="A",name="A counter",feature_definition_id="F_counter",
                                 role="COUNTER",comparison="GREATER_OR_EQUAL",threshold=1,threshold_unit="1")]
        features=[FeatureResult(feature_result_id="FR_"+d.feature_definition_id,feature_definition_id=d.feature_definition_id,
                                 run_id=ep.run_id,paired_nominal_run_id=ep.paired_nominal_run_id,episode_id=ep.episode_id,
                                 resolved_field=d.source_field_template,component_instance="sensor",value=v,unit=d.unit,
                                 window_start_s=0,window_end_s=2,sample_count=0 if v is None else 3,
                                 algorithm_version="1",input_data_sha256="a"*64,
                                 quality_flags=["MISSING"] if v is None else []) for d,v in zip(defs,(a,b,counter))]
        by_id={f.feature_definition_id:f for f in features}
        facts=[]
        for d in edefs:
            f=evaluate_evidence(d,by_id[d.feature_definition_id],all_feature_results=features)
            if f.episode_id=="UNKNOWN":f=f.model_copy(update={"episode_id":ep.episode_id})
            facts.append(f)
        bundle=EpisodeEvidenceBundle(episode=ep,analysis_component_instance="sensor",domain_id="UNIT_TEST",
                                     feature_definitions=defs,feature_results=features,evidence_definitions=edefs,evidence_facts=facts)
        ranking=FaultRanking(ranking_id="data",episode_id=ep.episode_id,source="DATA_DRIVEN",producer_id="test-fixture",
                            producer_version="1",candidates=[RankedFault(fault_id=f,rank=i+1,score=1/(i+1),source="DATA_DRIVEN")
                                                             for i,f in enumerate(data_ids)])
        graph=tmp_path/"graph";graph.mkdir(exist_ok=True)
        with (graph/"nodes.csv").open('w',newline='') as f:
            w=csv.writer(f);w.writerow(["node_id","node_type","name","properties_json"])
            for fault in ('A','B'):w.writerow([fault,'FaultMode',fault,'{}'])
            for d in edefs:w.writerow([d.evidence_definition_id,'EvidenceDefinition',d.name,json.dumps(d.model_dump(mode='json'))])
        with (graph/"edges.csv").open('w',newline='') as f:
            w=csv.writer(f);w.writerow(["edge_id","source_id","relation","target_id","properties_json"])
            for d in edefs:w.writerow(['EDGE_'+d.evidence_definition_id,d.target_fault_id,
                                      'CONTRADICTED_BY' if d.role.value=='COUNTER' else 'REQUIRES_EVIDENCE',d.evidence_definition_id,'{}'])
        data=OAGInput(bundle=bundle,diagnostic_ranking=ranking,registered_fault_ids=sorted({'A','B',*data_ids}),decision_policy='evidence_aware')
        return data,graph
    return build
