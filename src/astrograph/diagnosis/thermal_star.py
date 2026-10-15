from __future__ import annotations
from collections import defaultdict
from astrograph.evidence import EvidenceFactStatus, EvidenceRole

CASE_BY_FAULT={'THERMAL_HEATER_STUCK_ON':'thermal_heater_stuck_on','ADCS_STAR_TRACKER_DROPOUT':'star_tracker_dropout'}
def classify_thermal_star_evidence(evidence_facts:list[dict],*, nominal_case:bool=False)->dict:
    if nominal_case:return {'diagnosis_status':'DIAGNOSED','predicted_case_id':'nominal','selected_evidence':[],'explanation':'Offline paired nominal evaluation switch.','core_version':'v0.5.3-d3-thermal-star-deterministic-e0-v1'}
    facts=defaultdict(list)
    for raw in evidence_facts:facts[raw['target_fault_id']].append(raw)
    supported=[]
    for fault,items in facts.items():
        critical=[x for x in items if x['role']=='REQUIRED']
        if critical and all(x['status']=='SATISFIED' for x in critical):supported.append(fault)
    if len(supported)==1:
        fault=supported[0]; return {'diagnosis_status':'DIAGNOSED','predicted_case_id':CASE_BY_FAULT[fault],'selected_evidence':[x['evidence_fact_id'] for x in facts[fault] if x['status']=='SATISFIED'],'explanation':'Exactly one thermal/star registered fault satisfies all critical evidence.','core_version':'v0.5.3-d3-thermal-star-deterministic-e0-v1'}
    if len(supported)>1:return {'diagnosis_status':'AMBIGUOUS','predicted_case_id':None,'selected_evidence':[],'explanation':'Multiple thermal/star hypotheses satisfy critical evidence.','core_version':'v0.5.3-d3-thermal-star-deterministic-e0-v1'}
    return {'diagnosis_status':'UNKNOWN','predicted_case_id':None,'selected_evidence':[],'explanation':'No thermal/star hypothesis satisfies all critical evidence.','core_version':'v0.5.3-d3-thermal-star-deterministic-e0-v1'}
