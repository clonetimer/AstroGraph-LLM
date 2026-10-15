"""Bounded, audit-friendly multi-hop *hypothesis review* over document KG.

Only independent expert approved, scope-bound edges are eligible for the
'approved' channel. Public NASA/ECSS seed remains unreviewed and never crosses
into the approval channel. Graph traversal is not causal proof: reversing an
edge is an associative lookup only, not an implied reverse causal statement.
No LLM may invent intermediate hops, observations, citations or flight actions.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from astrograph.integrations.material_graph_builder import ALLOWED_TRIPLES, _normal
from astrograph.integrations.result_log_review import SourceLine

SCHEMA_VERSION = 'astrograph-material-multihop-reasoning.v1'
MAX_HOPS = 3
MAX_PATHS = 32
MAX_GRAPH_EDGES = 10000
_NEGATIVE = re.compile(r'(?i)\b(?:no|not|normal|nominal|cleared|resolved|recovered)\b|未出现|未发现|无异常|正常|恢复|已排除|已解除|否认|不存在')
_SHA = re.compile(r'^[0-9a-f]{64}$')
_SAFE_PUBLIC_STATE = 'PUBLIC_REFERENCE_SEED_UNREVIEWED'
_APPROVED_STATE = 'INDEPENDENTLY_REVIEWED_REFERENCE_RELATION'


def _path_id(layer: str, edges: list[dict[str, Any]], directions: list[str]) -> str:
    keys = [f"{e['edge_id']}:{d}" for e, d in zip(edges, directions)]
    return 'MHP-' + hashlib.sha256((layer + '|' + '|'.join(keys)).encode()).hexdigest()[:20].upper()


def _scoped_approved(edges: Sequence[Mapping[str, Any]], equipment_scope: str) -> tuple[list[dict], list[dict]]:
    """Fail-closed eligibility check; audit blocked claims separately."""
    eligible, blocked = [], []
    seen: set[str] = set()
    for edge in edges[:MAX_GRAPH_EDGES + 1]:
        edge_id = str(edge.get('proposal_id') or '')
        reasons = []
        src, dst = edge.get('source') or {}, edge.get('target') or {}
        triple = (src.get('entity_type'), edge.get('relation_type'), dst.get('entity_type'))
        prov = edge.get('provenance') or {}
        if edge.get('equipment_scope') != equipment_scope:
            reasons.append('MODEL_SCOPE_MISMATCH')
        if edge.get('review_status') != _APPROVED_STATE or not edge.get('reviewer') or not edge.get('truth_source_ref'):
            reasons.append('NOT_INDEPENDENTLY_APPROVED')
        if triple not in ALLOWED_TRIPLES or not src.get('entity_id') or not dst.get('entity_id'):
            reasons.append('ONTOLOGY_OR_ENDPOINT_INVALID')
        if not edge_id or edge_id in seen:
            reasons.append('MISSING_OR_DUPLICATE_RELATION_ID')
        if not edge.get('evidence_quote') or not _SHA.fullmatch(str(prov.get('source_sha256') or '')) or not prov.get('source_locators'):
            reasons.append('EVIDENCE_PROVENANCE_INCOMPLETE')
        if edge.get('polarity') not in {'AFFIRMATIVE', 'NEGATED'}:
            reasons.append('INVALID_POLARITY')
        if edge.get('claim_modality') not in {'REFERENCE_ASSERTION_REQUIRES_REVIEW', 'HYPOTHETICAL_OR_CONDITIONAL'}:
            reasons.append('UNKNOWN_MODALITY')
        if reasons:
            blocked.append({'edge_id': edge_id or 'UNKNOWN', 'reasons': reasons})
            continue
        seen.add(edge_id)
        eligible.append({'edge_id': edge_id, 'source': dict(src), 'target': dict(dst),
                         'relation_type': edge['relation_type'], 'polarity': edge['polarity'],
                         'claim_modality': edge['claim_modality'], 'provenance': dict(prov),
                         'evidence_quote': str(edge['evidence_quote'])[:1000],
                         'reviewer': edge['reviewer'], 'truth_source_ref': edge['truth_source_ref'],
                         'equipment_scope': equipment_scope, 'snapshot': edge.get('snapshot'),
                         'knowledge_layer': 'EXPERT_APPROVED_MODEL_BOUND'})
    if len(edges) > MAX_GRAPH_EDGES:
        blocked.append({'edge_id': 'CAPACITY', 'reasons': ['GRAPH_TOO_LARGE_FAIL_CLOSED']})
        return [], blocked
    return eligible, blocked


def _public_reference(graph_path: Path | None, fault_id: str, fault_name: str) -> dict[str, Any]:
    """Exact identifier/title matching only; no fuzzy model-name mapping."""
    if graph_path is None or not graph_path.is_file():
        return {'status': 'NOT_INSTALLED', 'edges': [], 'seed_ids': [], 'matched_fault_codes': []}
    try:
        if graph_path.stat().st_size > 4_000_000:
            raise ValueError('public graph exceeds bounded size')
        graph = json.loads(graph_path.read_text(encoding='utf-8'))
        if graph.get('graph_kind') != 'UNREVIEWED_PUBLIC_REFERENCE_SEED' or graph.get('is_flight_qualified') is not False:
            raise ValueError('not an unreviewed public seed')
        entities = {node['id']: node for node in graph['entities']}
        requested = {_normal(fault_id), _normal(fault_name)}
        clusters = [c for c in graph.get('clusters', [])
                    if _normal(c['code']) in requested or _normal(c['title']) in requested]
        seed_ids = [c['source_ids']['fault'] for c in clusters]
        edge_ids = {e for c in clusters for e in c['edge_ids']}
        # Only selected cluster edges; cross-subsystem candidates remain linked to their original fault.
        edges = []
        for relation in graph['relations']:
            if relation['id'] not in edge_ids:
                continue
            if (relation.get('review_status') != _SAFE_PUBLIC_STATE or
                relation.get('evidence_form') != 'CHINESE_PARAPHRASE_NOT_VERBATIM' or
                relation.get('automatic_control_authorized') is not False):
                raise ValueError('public edge classification mismatch')
            src, dst = entities[relation['source']], entities[relation['target']]
            if (src['entity_type'], relation['relation_type'], dst['entity_type']) not in ALLOWED_TRIPLES:
                raise ValueError('public graph ontology mismatch')
            edges.append({'edge_id':relation['id'], 'source': src, 'target':dst,
                          'relation_type':relation['relation_type'], 'polarity':'AFFIRMATIVE',
                          'claim_modality':'PUBLIC_PARAPHRASE_NOT_VERIFIED',
                          'provenance':{'source_ids':relation['source_ids'],
                                        'source_locator':relation['source_locator'],
                                        'evidence_form':relation['evidence_form']},
                          'evidence_quote':None, 'source_summary_zh':relation['source_summary_zh'],
                          'knowledge_layer':'UNREVIEWED_PUBLIC_REFERENCE_SEED',
                          'interpretation':relation.get('interpretation','')})
        return {'status':'MATCHED_UNREVIEWED_ONLY' if clusters else 'NO_EXACT_FAULT_MATCH',
                'edges':edges, 'seed_ids':seed_ids,
                'matched_fault_codes':[c['code'] for c in clusters]}
    except (ValueError, OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        return {'status':'INVALID_OR_UNAVAILABLE','edges':[],'seed_ids':[],
                'matched_fault_codes':[],'error_type':type(exc).__name__}


def _graph_paths(edges: Sequence[dict[str, Any]], seed_ids: set[str], *, layer: str,
                 max_hops: int = 3, max_paths: int = MAX_PATHS) -> list[dict[str, Any]]:
    """Enumerate bounded simple paths; record any inverse hop explicitly."""
    if not 1 <= max_hops <= MAX_HOPS:
        raise ValueError('maximum graph reasoning depth is 3')
    adjacency = defaultdict(list)
    for edge in sorted(edges, key=lambda item:item['edge_id']):
        src = edge['source'].get('entity_id') or edge['source'].get('id')
        dst = edge['target'].get('entity_id') or edge['target'].get('id')
        if not src or not dst or src == dst:
            continue
        adjacency[src].append((dst, edge, 'FORWARD'))
        # Reverse traversal is only an associative query, never a causal claim.
        if edge['relation_type'] not in {'MITIGATED_BY', 'REQUIRES', 'CHECKED_BY'}:
            adjacency[dst].append((src, edge, 'REVERSE_ASSOCIATION_ONLY'))
    results = []
    # Enumerate a bounded pool before ranking; otherwise a high-degree action
    # branch can crowd out all the symptom-anchored paths.
    candidate_pool_limit = max(128, max_paths * 8)
    expansion_budget = 3000
    for seed in sorted(seed_ids):
        stack = [(seed, (seed,), [], [])]
        while stack and expansion_budget > 0 and len(results) < candidate_pool_limit:
            node, visited, chain, dirs = stack.pop()
            expansion_budget -= 1
            if chain:
                terminal_edge = chain[-1]
                terminal = terminal_edge['target'] if dirs[-1]=='FORWARD' else terminal_edge['source']
                # Do not treat an intermediate FAULT_MODE or an action as current observation.
                results.append({
                    'path_id':_path_id(layer, chain, dirs),
                    'hop_count':len(chain),'seed_entity_id':seed,
                    'terminal_entity':{'id':node,'name':terminal['name'],
                                       'entity_type':terminal['entity_type'],
                                       'aliases':terminal.get('aliases',[])},
                    'hops':[{'edge_id':e['edge_id'],'direction':d,'relation_type':e['relation_type'],
                             'source':{'id':e['source'].get('entity_id') or e['source'].get('id'),
                                       'name':e['source']['name'],'entity_type':e['source']['entity_type']},
                             'target':{'id':e['target'].get('entity_id') or e['target'].get('id'),
                                       'name':e['target']['name'],'entity_type':e['target']['entity_type']},
                             'provenance':e['provenance'],'evidence_quote':e['evidence_quote'],
                             'source_summary_zh':e.get('source_summary_zh'),
                             'claim_modality':e['claim_modality'],'polarity':e['polarity']}
                            for e,d in zip(chain,dirs)],
                    'layer':layer,
                    'contains_reverse_association':any(d != 'FORWARD' for d in dirs),
                    'causal_proof':False,
                    'automatic_control_authorized':False,
                })
            if len(chain) >= max_hops:
                continue
            for nxt,edge,direction in reversed(adjacency.get(node,[])):
                if nxt in visited:
                    continue
                stack.append((nxt,(*visited,nxt),[*chain,edge],[*dirs,direction]))
    return sorted(results, key=lambda x:(
        x['terminal_entity']['entity_type'] != 'SYMPTOM',
        x['contains_reverse_association'],
        -x['hop_count'],x['path_id']))[:max_paths]


def _log_observations(lines: Sequence[SourceLine], symptom: Mapping[str, Any]) -> dict[str, Any]:
    """Exact label/alias string cues only; log event == unverified document text."""
    terms = [str(x).casefold() for x in [symptom.get('name'), *(symptom.get('aliases') or [])]
             if x and len(str(x).strip()) >= 4]
    positive, negative = [], []
    for index,line in enumerate(lines[:3000],1):
        if not any(term in line.text.casefold() for term in terms):
            continue
        receipt = {'evidence_id':f'MLOG-{index:05d}', 'filename':line.filename,
                   'locator':line.locator,'sha256':line.sha256, 'excerpt':line.text[:420],
                   'time_alignment_verified':False}
        (negative if _NEGATIVE.search(line.text) else positive).append(receipt)
        if len(positive) + len(negative) >= 12:
            break
    return {'supporting':positive,'contradicting':negative,
            'lexical_match_only':True,'sensor_observation_verified':False}


def evaluate_bidirectional_hypotheses(*, candidates: Sequence[Mapping[str, Any]],
                                      equipment_scope: str | None,
                                      log_lines: Sequence[SourceLine],
                                      reviewed_edges: Sequence[Mapping[str, Any]] = (),
                                      snapshot: int | None = None,
                                      max_hops: int = 3) -> dict[str, Any]:
    """Forward expectations + observed-symptom reverse *retrieval*, without inverse causality.

    This is intentionally a bounded shadow comparison. Model-released edges,
    negative relations, mixed-scope claims, conditional relations and paths
    containing reverse association never become positive expectations. Textual
    telemetry cues are not independently validated sensor observations.
    """
    if not 1 <= max_hops <= MAX_HOPS:
        raise ValueError('maximum graph reasoning depth is 3')
    if not 1 <= len(candidates) <= 12:
        raise ValueError('one to twelve competing candidates required')
    scope = str(equipment_scope or 'UNKNOWN').upper()
    result: dict[str, Any] = {
        'schema_version': 'astrograph-bidirectional-graph-review.v1',
        'status': 'ABSTAIN_NO_APPROVED_PATH', 'equipment_scope': scope,
        'snapshot': snapshot, 'candidates': [], 'reverse_observation_index': [],
        'shared_symptoms': [], 'blocked_edge_count': 0,
        'current_root_cause_proven': False, 'automatic_control_authorized': False,
        'claim_boundary': 'REVIEWED_REFERENCE_PATHS_AND_UNVERIFIED_TEXT_CUES_ONLY',
        'diagnostic_vote_allowed': False,
    }
    if scope in {'UNKNOWN', 'UNSPECIFIED'} or not re.fullmatch(r'[A-Z0-9][A-Z0-9_.:-]{1,79}',scope):
        result['status'] = 'ABSTAIN_EQUIPMENT_SCOPE_REQUIRED'
        result['candidates'] = [{
            'fault_id': str(row.get('fault_id') or ''),
            'fault_name': str(row.get('fault_name') or ''),
            'expected_symptoms': [], 'forward_path_count': 0,
            'validated_root_cause': False,
            'status': 'ABSTAIN_EQUIPMENT_SCOPE_REQUIRED',
        } for row in candidates]
        return result
    approved, blocked = _scoped_approved(reviewed_edges, scope)
    # A pinned graph read may carry earlier snapshot references; if the caller
    # supplies a snapshot, never admit edges bound to a different one.
    scoped = [e for e in approved if snapshot is None or e.get('snapshot') is None
              or (type(e['snapshot']) is int and 0 <= e['snapshot'] <= snapshot)]
    result['blocked_edge_count'] = len(blocked) + len(approved) - len(scoped)
    signed = defaultdict(set)
    for e in scoped:
        if e['claim_modality'] != 'REFERENCE_ASSERTION_REQUIRES_REVIEW':
            continue
        signed[(e['source']['entity_id'],e['relation_type'],
                e['target']['entity_id'])].add(e['polarity'])
    # Do not build paths through a contradicting edge and then only reject the
    # last hop. A conflict anywhere on a route makes the route unusable.
    usable = [e for e in scoped if e['polarity'] == 'AFFIRMATIVE'
              and e['claim_modality'] == 'REFERENCE_ASSERTION_REQUIRES_REVIEW'
              and signed[(e['source']['entity_id'],e['relation_type'],
                          e['target']['entity_id'])] == {'AFFIRMATIVE'}]
    seen_faults: set[str] = set()
    observations: dict[str, dict[str, Any]] = {}
    symptom_to_faults: dict[str, set[str]] = defaultdict(set)
    for item in candidates:
        fault_id = str(item.get('fault_id') or '')
        fault_name = str(item.get('fault_name') or '')
        if not fault_id or fault_id in seen_faults:
            raise ValueError('candidate IDs must be nonempty and unique')
        seen_faults.add(fault_id)
        seeds = set()
        labels = {_normal(fault_id), _normal(fault_name)}
        for edge in usable:
            for node in (edge['source'], edge['target']):
                if node.get('entity_type') == 'FAULT_MODE' and labels.intersection(
                    {_normal(str(x)) for x in [node.get('name'),
                                               *(node.get('aliases') or [])] if x}):
                    seeds.add(node['entity_id'])
        paths = _graph_paths(usable, seeds, layer='REVIEWED_FORWARD_EXPECTATION',
                             max_hops=max_hops, max_paths=MAX_PATHS)
        valid = [path for path in paths
                 if path['terminal_entity']['entity_type'] == 'SYMPTOM'
                 and not path['contains_reverse_association']
                 and all(hop['direction'] == 'FORWARD' for hop in path['hops'])]
        # Multiple paths to one symptom are not independent confirmations.
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for path in valid:
            grouped[path['terminal_entity']['id']].append(path)
        symptom_rows = []
        for symptom_id in sorted(grouped):
            path_group = grouped[symptom_id]
            symptom = path_group[0]['terminal_entity']
            cues = _log_observations(log_lines, symptom)
            reference_sources = sorted({hop['provenance']['source_sha256']
                  for path in path_group for hop in path['hops']})
            row = {'symptom_id': symptom_id, 'symptom_name': symptom['name'],
                   'path_ids': [path['path_id'] for path in path_group],
                   'reference_source_sha256': reference_sources,
                   'supporting_log_cue_ids': [v['evidence_id'] for v in cues['supporting']],
                   'contradicting_log_cue_ids': [v['evidence_id'] for v in cues['contradicting']],
                   'sensor_observation_verified': False,
                   'matching_semantics': 'EXACT_TEXT_CUE_NOT_SENSOR_VALIDATION'}
            symptom_rows.append(row)
            symptom_to_faults[symptom_id].add(fault_id)
            if cues['supporting'] or cues['contradicting']:
                entry = observations.setdefault(symptom_id, {
                    'symptom_id': symptom_id, 'symptom_name': symptom['name'],
                    'supporting_log_cue_ids': row['supporting_log_cue_ids'],
                    'contradicting_log_cue_ids': row['contradicting_log_cue_ids'],
                    'candidate_fault_ids': set(),
                    'reverse_direction_is_causal': False})
                entry['candidate_fault_ids'].add(fault_id)
        result['candidates'].append({
            'fault_id': fault_id, 'fault_name': fault_name,
            'expected_symptoms': symptom_rows,
            'missing_symptom_cue_ids': [x['symptom_id'] for x in symptom_rows
                                        if not x['supporting_log_cue_ids']
                                        and not x['contradicting_log_cue_ids']],
            'forward_path_count': len(valid),
            'validated_root_cause': False,
        })
    result['shared_symptoms'] = [
        {'symptom_id': sid, 'candidate_fault_ids': sorted(ids)}
        for sid, ids in sorted(symptom_to_faults.items()) if len(ids) > 1]
    result['reverse_observation_index'] = [
        {**row, 'candidate_fault_ids': sorted(row['candidate_fault_ids'])}
        for _, row in sorted(observations.items())]
    if result['reverse_observation_index']:
        result['status'] = 'LEXICAL_REVERSE_POINTERS_REQUIRE_OBSERVATION_VALIDATION'
    elif any(item['expected_symptoms'] for item in result['candidates']):
        result['status'] = 'FORWARD_EXPECTATIONS_ONLY_NO_OBSERVATION_CUE'
    return result


def evaluate_multihop(*, fault_id: str, fault_name: str, equipment_scope: str | None,
                      log_lines: Sequence[SourceLine], reviewed_edges: Sequence[Mapping[str,Any]] = (),
                      snapshot: int | None = None, public_graph_path: Path | None = None,
                      max_hops: int = 3) -> dict[str, Any]:
    """GraphRAG evidence chain + fail-closed uncertainty/abstention contract."""
    if not 1 <= max_hops <= MAX_HOPS:
        raise ValueError('max_hops must be between 1 and 3')
    scope = str(equipment_scope or 'UNKNOWN').upper()
    valid_scope = scope not in {'UNKNOWN','UNSPECIFIED'} and bool(re.fullmatch(r'[A-Z0-9][A-Z0-9_.:-]{1,79}',scope))
    approved, blocked = _scoped_approved(reviewed_edges,scope) if valid_scope else ([],[])
    by_key=defaultdict(set)
    for e in approved:
        by_key[(e['source']['entity_id'],e['relation_type'],e['target']['entity_id'])].add(e['polarity'])
    conflict_keys={key for key,values in by_key.items() if len(values)>1}
    node_seeds=set()
    for e in approved:
        for node in (e['source'],e['target']):
            if (node['entity_type']=='FAULT_MODE' and
                {_normal(fault_id),_normal(fault_name)}.intersection(
                    {_normal(n) for n in [node['name'],*node.get('aliases',[])]})):
                node_seeds.add(node['entity_id'])
    approved_paths=_graph_paths(approved,node_seeds,layer='EXPERT_APPROVED',max_hops=max_hops)
    valid_paths=[]
    rejected_paths=[]
    observation_support=[]
    observation_oppose=[]
    relevant_conflicts=set()
    for path in approved_paths:
        relevant_conflicts.update((h['source']['id'],h['relation_type'],h['target']['id'])
                                  for h in path['hops']
                                  if (h['source']['id'],h['relation_type'],h['target']['id']) in conflict_keys)
        path['conflict_edge_ids']=[h['edge_id'] for h in path['hops']
            if (h['source']['id'],h['relation_type'],h['target']['id']) in conflict_keys]
        path['negated_edge_ids']=[h['edge_id'] for h in path['hops'] if h['polarity'] != 'AFFIRMATIVE']
        path['conditional_edge_ids']=[h['edge_id'] for h in path['hops']
                                      if h['claim_modality']=='HYPOTHETICAL_OR_CONDITIONAL']
        path['eligible_as_hypothesis_path'] = not (path['conflict_edge_ids'] or path['negated_edge_ids']
                or path['conditional_edge_ids'] or path['contains_reverse_association'])
        obs=(_log_observations(log_lines,path['terminal_entity'])
             if path['terminal_entity']['entity_type']=='SYMPTOM' else
             {'supporting':[], 'contradicting':[], 'lexical_match_only':True,'sensor_observation_verified':False})
        path['log_crosscheck']=obs
        if path['eligible_as_hypothesis_path']:
            valid_paths.append(path)
            observation_support.extend(obs['supporting'])
            observation_oppose.extend(obs['contradicting'])
        else:
            rejected_paths.append({'path_id':path['path_id'],
              'reasons':([*(['CONFLICTED_KNOWLEDGE'] if path['conflict_edge_ids'] else []),
                         *(['NEGATED_CLAIM'] if path['negated_edge_ids'] else []),
                         *(['CONDITIONAL_CLAIM'] if path['conditional_edge_ids'] else []),
                         *(['REVERSE_RELATION_NOT_CAUSAL'] if path['contains_reverse_association'] else [])])})
    pub = _public_reference(public_graph_path,fault_id,fault_name)
    public_paths=_graph_paths(pub['edges'],set(pub['seed_ids']),layer='UNREVIEWED_PUBLIC_SEED',
                             max_hops=max_hops,max_paths=12)
    for path in public_paths:
        path['eligible_as_hypothesis_path']=False
        path['review_status']='UNREVIEWED_DO_NOT_USE_FOR_DIAGNOSIS'
    # A rejected or negated symptom does not magically become evidence for an alternative cause.
    unique_support={e['evidence_id']:e for e in observation_support}
    unique_oppose={e['evidence_id']:e for e in observation_oppose}
    if not valid_scope:
        decision='ABSTAIN_EQUIPMENT_SCOPE_REQUIRED'
    elif unique_oppose:
        decision='ABSTAIN_CONTRADICTORY_LOG_EVIDENCE'
    elif unique_support and valid_paths:
        decision='PROVISIONAL_EVIDENCE_BACKED_HYPOTHESIS'
    elif not approved_paths:
        decision='ABSTAIN_NO_APPROVED_PATH'
    elif not valid_paths:
        decision='ABSTAIN_GRAPH_PATH_BLOCKED'
    else:
        decision='ABSTAIN_NO_SYMPTOM_LOG_ANCHOR'
    gap_codes=[]
    if not valid_scope:
        gap_codes.append('MISSING_EQUIPMENT_SCOPE')
    if not approved_paths:
        gap_codes.append('MISSING_APPROVED_FAULT_RELATIONS')
    if blocked:
        gap_codes.append('INVALID_APPROVED_RELATION_PROVENANCE')
    if rejected_paths:
        gap_codes.append('CONFLICTED_NEGATED_CONDITIONAL_OR_REVERSE_PATH')
    if unique_oppose:
        gap_codes.append('LOG_CONTRADICTION_REQUIRES_REVIEW')
    if not unique_support:
        gap_codes.append('MISSING_INDEPENDENT_SYMPTOM_OBSERVATION')
    if pub['status']=='MATCHED_UNREVIEWED_ONLY' and not valid_paths:
        gap_codes.append('PUBLIC_SEED_REQUIRES_MODEL_SPECIFIC_REVIEW')
    # Classification is deliberately not a numerical probability and cannot authorize commands.
    return {
        'schema_version':SCHEMA_VERSION,'fault_id':fault_id,'fault_name':fault_name,
        'equipment_scope':scope,'graph_snapshot':snapshot,
        'max_hops':max_hops,'decision':decision,
        'response_mode':('EVIDENCE_BOUND_PROVISIONAL_ONLY' if decision.startswith('PROVISIONAL')
                         else 'ABSTAIN_FROM_ROOT_CAUSE'),
        'approved_paths':approved_paths, 'public_reference_paths':public_paths,
        'approved_path_count':len(approved_paths),
        'eligible_hypothesis_path_count':len(valid_paths),
        'rejected_path_reasons':rejected_paths,
        'blocked_claims':blocked[:24], 'knowledge_conflict_groups':len(relevant_conflicts),
        'independent_log_support_cues':list(unique_support.values()),
        'independent_log_opposition_cues':list(unique_oppose.values()),
        'public_seed_lookup':{k:v for k,v in pub.items() if k not in ('edges','seed_ids')},
        'knowledge_gap_codes':gap_codes,
        'requires_expert_review':True,'verified_current_root_cause':False,
        'numeric_confidence_available':False,'automatic_control_authorized':False,
        'no_inferred_missing_hops':True,
        'explanation':('仅形成待验证的证据支持假设，不代表根因已确诊；日志匹配仍需时标和传感器核验。'
                       if decision.startswith('PROVISIONAL') else
                       '当前证据链不能支持可靠的根因判断；不得由语言模型补造缺失的关系或处置指令。'),
    }


def default_public_graph_path() -> Path:
    return Path(__file__).resolve().parents[3] / 'reference' / 'public_spacecraft_knowledge_seed' / 'graph.json'


def apply_material_answer_guard(knowledge: dict[str,Any], reasoning: dict[str,Any]) -> dict[str,Any]:
    """Protect the user-facing material answer from citation-only LLM hallucinations.

    An auditable digest replaces the generated narrative in the *public report*;
    any raw generation receipt stays only in the host-side knowledge artifact
    written by the enrichment service, not this public JSON/Markdown report.
    Valid citation IDs never certify the semantic truth of a generated claim.
    Unlike legacy telemetry tasks, material reviews *always* require a
    deterministic epistemic status from the multi-hop evaluator.
    """
    all_abstained=bool((reasoning.get('answer_policy') or {}).get('all_candidates_abstained',True))
    decisions=[x.get('multi_hop',{}).get('decision','NOT_EVALUATED')
               for x in (reasoning.get('candidates') or [])]
    if all_abstained:
        summary='当前型号级已审核知识与日志无法形成可核验的故障原因证据链；无法确定根因，请补充资料并交由专家复核。'
    else:
        summary='已找到部分与日志线索相关的已审知识路径，只能形成待验证的原因假设；不构成实际故障确诊或操作授权。'
    output=dict(knowledge)
    # Web/API report JSON is visible to operators. Do not simply null
    # generation.explanation and leave an open-ended generation dictionary:
    # parsing errors or provider receipts sometimes contain the raw model
    # response, including unverifiable claims and injected text. Use an
    # allow-list. The original generation receipt remains in the private
    # server-side knowledge artifact for authorized investigation.
    raw_generation=dict(output.get('generation') or {})
    generated={
        'status':str(raw_generation.get('status') or 'NOT_CONFIGURED'),
        'explanation':None,
        'semantic_hallucination_assessment':'NOT_EVALUATED',
    }
    if raw_generation.get('citation_reference_validation') in {'PASS','FAILED','NOT_EVALUATED'}:
        generated['citation_reference_validation']=raw_generation['citation_reference_validation']
    raw_explanation=raw_generation.get('explanation')
    model_explanation=None
    if raw_explanation is not None:
        serialized=json.dumps(raw_explanation,sort_keys=True,ensure_ascii=False,
                              separators=(',',':'),default=str)
        generated['unverified_explanation_sha256']=hashlib.sha256(serialized.encode('utf-8')).hexdigest()
        generated['unverified_explanation_withheld_from_report']=True
        generated['raw_audit_location']='SERVER_SIDE_JOB_KNOWLEDGE_ARTIFACT_NOT_PUBLIC_REPORT'
        # The enrichment service has already schema-validated the object and checked
        # every citation identifier. Expose only this narrow allow-list; provider
        # receipts, prompts and unstructured error text remain private.
        if (raw_generation.get('status') == 'SUCCEEDED'
                and raw_generation.get('citation_reference_validation') == 'PASS'
                and isinstance(raw_explanation,dict)):
            from astrograph.retrieval.material_claim_guard import audit_material_claims
            claim_audit = audit_material_claims(output, reasoning, raw_explanation)
            safe_claims = claim_audit['published_extracts']
            generated['claim_level_audit'] = {
                key: value for key, value in claim_audit.items() if key != 'published_extracts'
            }
            model_explanation={
                'status':'EXTRACTIVE_REVIEW_ONLY' if safe_claims else 'NO_VERIFIABLE_EXTRACTS',
                'source':'LLM_EXTRACTIVE_REVIEW',
                'summary':(
                    f"已比对 {len(claim_audit['claim_receipts'])} 条模型声明，"
                    f"仅保留 {len(safe_claims)} 条来源原文完全匹配的摘录；"
                    '文字一致不等于证明因果关系。'
                ),
                'candidate_fault_id':(
                    raw_explanation.get('candidate_fault_id')
                    if raw_explanation.get('candidate_fault_id') in {
                        c.get('fault_id') for c in reasoning.get('candidates', [])
                        if isinstance(c, dict)
                    } else None
                ),
                'abstained':bool(raw_explanation.get('abstained')),
                'claims':safe_claims,
                'recommended_review_actions':[],
                'limitations':['未独立验证原文的因果真值；原始生成内容仅存于服务端审计产物。'],
                'citation_reference_validation':str(raw_generation.get('citation_reference_validation') or 'NOT_EVALUATED'),
                'semantic_hallucination_assessment':'EXTRACTIVE_ONLY_NOT_ENTAILMENT',
                'automatic_control_authorized':False,
            }
            # Keep the unchecked model summary, hypotheses and free-form action
            # strings out of user-facing reports even when their IDs were valid.
            generated['unverified_explanation_withheld_from_report']=True
    if raw_generation.get('error') is not None:
        generated['error_details_withheld_from_report']=True
    if raw_generation.get('backend_receipt') is not None:
        generated['backend_receipt_withheld_from_report']=True
    output['generation']=generated
    output['model_explanation']=model_explanation or {
        'status':'NOT_AVAILABLE',
        'source':'RULE_ENGINE_ONLY',
        'automatic_control_authorized':False,
    }
    synthesis=dict(output.get('deterministic_synthesis') or {})
    if 'explanation' in synthesis:
        synthesis['explanation']=None
        synthesis['material_synthesis_withheld_from_report']=True
        output['deterministic_synthesis']=synthesis
    output['material_answer_guard']={
        'status':'ABSTAIN' if all_abstained else 'PROVISIONAL_ONLY',
        'decisions':decisions, 'unverified_generated_narrative_exposed':False,
        'validated_structured_model_output_exposed':bool(model_explanation and model_explanation.get('claims')),
        'citation_id_validation_is_not_semantic_verification':True,
        'automatic_control_authorized':False,
        'claim_boundary':'A missing graph edge, negation or unverified reference cannot be filled by an LLM.',
    }
    output['final_explanation']={
        'status':'EPISTEMIC_GUARD_APPLIED',
        'source':'DETERMINISTIC_MATERIAL_MULTIHOP_POLICY',
        'explanation':{'summary':summary,'claims':[]},
        'citation_reference_validation':'NOT_A_CAUSAL_CERTIFICATION',
        'semantic_hallucination_assessment':'UNVERIFIED_NARRATIVE_WITHHELD',
    }
    return output
