"""ADAPT mechanism-model registry for governed simulation-diagnosis.

R30 generalizes the single R29 lamp provider into an explicit component/fault
mechanism catalogue.  A registry entry declares what is physically modeled,
what telemetry is expected to respond, and whether AstroGraph currently has an
executable numerical provider.  Non-executable entries are still useful for
coverage/governance and must not be represented as simulated trajectories.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict, deque
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _digest(value: Any) -> str:
    raw=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


class MechanismModelClass(str, Enum):
    RESISTIVE_LOAD_FAILED_OFF = "RESISTIVE_LOAD_FAILED_OFF"
    SWITCH_STATE_TRANSITION = "SWITCH_STATE_TRANSITION"
    ROTATING_MACHINE_SPEED_RESPONSE = "ROTATING_MACHINE_SPEED_RESPONSE"
    PUMP_FLOW_RESPONSE = "PUMP_FLOW_RESPONSE"
    INVERTER_OUTPUT_COLLAPSE = "INVERTER_OUTPUT_COLLAPSE"
    SENSOR_MEASUREMENT_FAULT = "SENSOR_MEASUREMENT_FAULT"
    BATTERY_DEGRADATION = "BATTERY_DEGRADATION"
    PASSIVE_LOAD_FAILED_OFF = "PASSIVE_LOAD_FAILED_OFF"


class ModelFidelity(str, Enum):
    NUMERICAL_REDUCED_ORDER = "NUMERICAL_REDUCED_ORDER"
    QUALITATIVE_STATE_RESPONSE = "QUALITATIVE_STATE_RESPONSE"
    DECLARED_ONLY_NOT_EXECUTABLE = "DECLARED_ONLY_NOT_EXECUTABLE"


class ADAPTMechanismModelSpec(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True,allow_inf_nan=False)
    model_id:str=Field(min_length=1)
    component_id:str=Field(min_length=1)
    component_type:str=Field(min_length=1)
    fault_mode:str=Field(min_length=1)
    fault_id:str=Field(min_length=1)
    model_class:MechanismModelClass
    fidelity:ModelFidelity
    executable_provider_id:str|None=None
    state_transition:str=Field(min_length=1)
    expected_response_codes:tuple[str,...]=()
    observable_component_ids:tuple[str,...]=()
    command_ids:tuple[str,...]=()
    parameter_source:str=Field(min_length=1)
    truth_independent:bool=True
    formal_evidence_eligible:bool=False

    @model_validator(mode='after')
    def valid(self):
        if self.fidelity==ModelFidelity.NUMERICAL_REDUCED_ORDER and not self.executable_provider_id:
            raise ValueError('NUMERICAL_MODEL_REQUIRES_PROVIDER')
        if not self.truth_independent or self.formal_evidence_eligible:
            raise ValueError('R30_MECHANISM_MODEL_GOVERNANCE_VIOLATION')
        return self


class ADAPTMechanismModelRegistry(BaseModel):
    model_config=ConfigDict(extra='forbid',frozen=True)
    schema_version:str='astrograph-adapt-mechanism-model-registry.v1'
    source_catalog_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    models:tuple[ADAPTMechanismModelSpec,...]
    coverage:dict[str,Any]
    registry_sha256:str=Field(pattern=r'^[0-9a-f]{64}$')
    claim_boundary:str=(
        'Mechanism-model coverage catalogue. Numerical execution exists only where executable_provider_id is set. '
        'Qualitative/declarative entries are not simulated trajectories, calibrated probabilities, formal EvidenceFacts, or release authority.'
    )

    @model_validator(mode='after')
    def verify(self):
        payload={'source_catalog_sha256':self.source_catalog_sha256,'models':[x.model_dump(mode='json') for x in self.models],'coverage':self.coverage}
        if _digest(payload)!=self.registry_sha256: raise ValueError('ADAPT_MECHANISM_REGISTRY_SHA_MISMATCH')
        ids=[x.model_id for x in self.models]
        if len(ids)!=len(set(ids)): raise ValueError('DUPLICATE_MECHANISM_MODEL_ID')
        return self


def _catalog_sha(catalog:Mapping[str,Any])->str:
    return _digest(catalog)


def _maps(catalog:Mapping[str,Any]):
    components={str(x['component_id']):x for x in catalog.get('components',[])}
    adj:dict[str,set[str]]=defaultdict(set)
    for e in catalog.get('connections',[]):
        a=str(e.get('c1') or '');b=str(e.get('c2') or '')
        if a and b: adj[a].add(b);adj[b].add(a)
    return components,adj


def _nearby(components,adj,start,predicate,max_hops=3):
    q=deque([(start,0)]);seen={start};out=[]
    while q:
        n,d=q.popleft()
        if d and predicate(components.get(n,{})): out.append((d,n))
        if d>=max_hops: continue
        for m in sorted(adj.get(n,set())):
            if m not in seen: seen.add(m);q.append((m,d+1))
    return tuple(n for _,n in sorted(out))


def build_adapt_mechanism_model_registry(catalog:Mapping[str,Any])->ADAPTMechanismModelRegistry:
    components,adj=_maps(catalog)
    models=[]
    for cid,c in sorted(components.items()):
        ctype=str(c.get('component_type') or '')
        modes=tuple(sorted(set(map(str,c.get('fault_modes',[])))))
        commands=tuple(sorted(set(map(str,c.get('command_ids',[])))))
        sensors=_nearby(components,adj,cid,lambda x:x.get('kind')=='sensor',3)
        for mode in modes:
            fault_id=f'{cid}:{mode}'
            mclass=None;fidelity=ModelFidelity.QUALITATIVE_STATE_RESPONSE;provider=None;transition='FAULT_MODE_APPLIED';responses=[];parameter='ADAPT_XML_TOPOLOGY_AND_FAULT_MODE'
            if re.fullmatch(r'LightBulb[0-9]+W',ctype) and mode=='FailedOff':
                mclass=MechanismModelClass.RESISTIVE_LOAD_FAILED_OFF
                fidelity=ModelFidelity.NUMERICAL_REDUCED_ORDER
                provider='adapt-resistive-lightbulb-reduced-order-v1'
                transition='ENERGIZED_TO_OPEN_LOAD_ZERO_BRANCH_POWER'
                responses=['BRANCH_CURRENT_DECREASE','LOCAL_TEMPERATURE_COOLING','LIGHT_OUTPUT_LOSS']
                parameter='COMPONENT_TYPE_RATED_WATTS_PLUS_HOST_BOUND_BRANCH_VOLTAGE_CURRENT'
            elif ctype in {'Relay','CircuitBreaker70AmpCmd','CircuitBreaker70Amp','CircuitBreaker10Amp','CircuitBreaker4Amp','PositionSensor'}:
                mclass=MechanismModelClass.SWITCH_STATE_TRANSITION
                transition=f'DISCRETE_SWITCH_OR_POSITION_MODE_{mode.upper()}'
                responses=['POSITION_STATE_CONSISTENCY','DOWNSTREAM_ENERGIZATION_CHANGE']
            elif ctype in {'LargeFan','SmallFan'}:
                mclass=MechanismModelClass.ROTATING_MACHINE_SPEED_RESPONSE
                transition=f'ROTATING_MACHINE_MODE_{mode.upper()}'
                responses=['SPEED_RESPONSE','CURRENT_RESPONSE','DOWNSTREAM_AIRFLOW_OR_THERMAL_RESPONSE']
            elif ctype=='WaterPump':
                mclass=MechanismModelClass.PUMP_FLOW_RESPONSE
                if mode in {'FailedOff','FlowBlocked'}:
                    fidelity=ModelFidelity.NUMERICAL_REDUCED_ORDER
                    provider='adapt-waterpump-reduced-order-v1'
                    parameter='ADAPT_XML_WATERPUMP_NOMINAL_WATTS_FLOW_PLUS_HOST_BOUND_CURRENT_VOLTAGE_FLOW'
                transition=f'PUMP_MODE_{mode.upper()}'
                responses=['FLOW_RESPONSE','CURRENT_RESPONSE']
            elif ctype=='Inverter' and mode=='FailedOff':
                mclass=MechanismModelClass.INVERTER_OUTPUT_COLLAPSE
                transition='AC_OUTPUT_ENERGIZED_TO_OFF'
                responses=['AC_VOLTAGE_COLLAPSE','AC_FREQUENCY_LOSS','DOWNSTREAM_CURRENT_CHANGE']
            elif c.get('kind')=='sensor' and mode in {'Offset','Stuck'}:
                mclass=MechanismModelClass.SENSOR_MEASUREMENT_FAULT
                transition=f'MEASUREMENT_CHANNEL_{mode.upper()}'
                responses=['MEASUREMENT_ONLY_CHANGE','PHYSICAL_NEIGHBOR_NONRESPONSE_EXPECTED']
            elif ctype.startswith('Battery') and mode=='Degraded':
                mclass=MechanismModelClass.BATTERY_DEGRADATION
                fidelity=ModelFidelity.DECLARED_ONLY_NOT_EXECUTABLE
                transition='AVAILABLE_CAPACITY_OR_TERMINAL_VOLTAGE_MARGIN_DECREASE'
                responses=['DC_VOLTAGE_DROOP_UNDER_LOAD','CAPACITY_MARGIN_DECREASE']
            elif ctype in {'Resistor'} and mode=='FailedOff':
                mclass=MechanismModelClass.PASSIVE_LOAD_FAILED_OFF
                fidelity=ModelFidelity.DECLARED_ONLY_NOT_EXECUTABLE
                transition='CONDUCTIVE_PATH_TO_OPEN'
                responses=['BRANCH_CURRENT_DECREASE']
            if mclass is None:
                continue
            models.append(ADAPTMechanismModelSpec(
                model_id=f'ADAPT-MECH-{cid}-{mode}',component_id=cid,component_type=ctype,fault_mode=mode,fault_id=fault_id,
                model_class=mclass,fidelity=fidelity,executable_provider_id=provider,state_transition=transition,
                expected_response_codes=tuple(responses),observable_component_ids=tuple(sensors),command_ids=commands,
                parameter_source=parameter,
            ))
    models=tuple(sorted(models,key=lambda x:x.model_id))
    fault_universe=sum(len(set(map(str,c.get('fault_modes',[])))) for c in components.values())
    numerical=sum(x.fidelity==ModelFidelity.NUMERICAL_REDUCED_ORDER for x in models)
    qualitative=sum(x.fidelity==ModelFidelity.QUALITATIVE_STATE_RESPONSE for x in models)
    declared=sum(x.fidelity==ModelFidelity.DECLARED_ONLY_NOT_EXECUTABLE for x in models)
    coverage={
        'catalog_component_count':len(components),
        'catalog_fault_mode_instance_count':fault_universe,
        'modeled_fault_mode_instance_count':len(models),
        'modeled_fault_mode_coverage':(len(models)/fault_universe if fault_universe else 0.0),
        'numerical_reduced_order_count':numerical,
        'qualitative_state_response_count':qualitative,
        'declared_only_count':declared,
        'numerical_reduced_order_coverage_of_fault_universe':(numerical/fault_universe if fault_universe else 0.0),
        'by_model_class':{k.value:sum(x.model_class==k for x in models) for k in MechanismModelClass},
    }
    source_sha=_catalog_sha(catalog)
    payload={'source_catalog_sha256':source_sha,'models':[x.model_dump(mode='json') for x in models],'coverage':coverage}
    return ADAPTMechanismModelRegistry(source_catalog_sha256=source_sha,models=models,coverage=coverage,registry_sha256=_digest(payload))

__all__=['ADAPTMechanismModelRegistry','ADAPTMechanismModelSpec','MechanismModelClass','ModelFidelity','build_adapt_mechanism_model_registry']
