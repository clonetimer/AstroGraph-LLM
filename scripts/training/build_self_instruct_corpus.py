"""Instantiate assistant-authored tasks against literal simulator declarations.

No LLM API is called: instruction authorship and deterministic instance binding
are recorded separately. Full source files are assigned before extraction.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
from astrograph.training.self_instruct import GeneratedExample, filter_tasks, prepare_generated

SPLITS = {
    'train': ['reaction_wheel', 'transmitter', 'imu', 'pdu', 'solar_panel', 'thermal_node', 'heater', 'radiator'],
    'validation': ['star_tracker', 'magnetometer'],
    'test': ['thruster', 'fuel_tank'],
}


def select(effects, mode):
    if mode == 'first': return effects[:1]
    if mode == 'last': return effects[-1:]
    def matches(e):
        field, op, value, _ = e['args']
        if mode in {'all', 'fields', 'mechanisms'}: return True
        if mode in {'add', 'multiply', 'override'}: return op == mode
        if mode == 'zero': return value == 0
        if mode == 'nonzero': return value != 0
        if mode == 'above_one': return value > 1
        if mode == 'negative': return value < 0
        if mode == 'fraction': return 0 < value < 1
        if mode == 'power': return field.endswith('_w')
        if mode == 'availability': return field == 'availability'
        if mode == 'not_availability': return field != 'availability'
        if mode == 'thermal': return any(x in field for x in ('heat', 'temperature', 'thermal', 'emissivity', 'absorptivity'))
        if mode == 'sensor': return any(x in field for x in ('bias', 'noise', 'error', 'dropout'))
        raise ValueError('unknown task mode: ' + mode)
    return [e for e in effects if matches(e)]


def build(source_root: Path, task_file: Path, output: Path):
    tasks = filter_tasks(json.loads(task_file.read_text())['tasks'])
    output.mkdir(parents=True, exist_ok=False)
    frozen = []
    for split, components in SPLITS.items():
        for component in components:
            path = Path('src/components') / component / 'faults.py'
            raw = (source_root / path).read_bytes()
            frozen.append({'split': split, 'component': component, 'path': path.as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()})
    (output / 'split-plan.json').write_text(json.dumps(frozen, indent=2) + '\n')
    examples = []
    source_propositions = set()
    for source in frozen:
        raw = (source_root / source['path']).read_text()
        lines = raw.splitlines(keepends=True)
        calls = sorted((n for n in ast.walk(ast.parse(raw)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                        and n.func.id == 'ParameterEffect' and len(n.args) == 4), key=lambda n: n.lineno)
        effects = []
        for n in calls:
            try: args = [ast.literal_eval(a) for a in n.args]
            except (ValueError, TypeError): continue
            if not isinstance(args[2], (int, float)): continue
            quote = ast.get_source_segment(raw, n)
            effects.append({'args': args, 'quote': quote, 'start': n.lineno, 'end': n.end_lineno})
            source_propositions.add((source['sha256'], quote))
        # Bound held-out volume before model execution; not selected by outcomes.
        chunks = [effects[i:i+2] for i in range(0, len(effects), 2)]
        if source['split'] != 'train': chunks = chunks[:2]
        for ci, chunk in enumerate(chunks):
            excerpt = ''.join(lines[chunk[0]['start']-1:chunk[-1]['end']])
            docid = 'SIM-' + hashlib.sha256((source['path'] + ':' + str(ci)).encode()).hexdigest()[:16]
            # Four deterministic task choices per excerpt and family; source order
            # rotates coverage. Held-out choices fixed independently of models.
            for family in ('ONTOLOGY_EXTRACTION', 'RETRIEVAL_QUERY'):
                family_tasks = [t for t in tasks['accepted'] if t['task'] == family]
                chosen = [family_tasks[(ci * 4 + j) % len(family_tasks)] for j in range(min(4, len(family_tasks)))]
                for task in chosen:
                    selected = select(chunk, task['mode'])
                    common = {'component': source['component'], 'evidence_scope': 'PROJECT_SIMULATION',
                              'association_rule': 'subject=literal mechanism description; predicate=associated_with; object=parameter identifier; source_quote=exact ParameterEffect call. No causal or flight claim.'}
                    if family == 'ONTOLOGY_EXTRACTION':
                        inputs = {**common, 'documents': [{'document_id': docid, 'text': excerpt}]}
                        expected = {'triples': [{'subject': e['args'][3], 'predicate': 'associated_with', 'object': e['args'][0],
                            'source_document_id': docid, 'source_quote': e['quote']} for e in selected],
                            'lifecycle': 'REVIEW_REQUIRED', 'active_kg_modified': False}
                    else:
                        inputs = {**common, 'source_ids': [docid], 'source_text': excerpt,
                                  'query_rule': 'component then selected mechanism phrases then field names, each once, in source order; fields/mechanisms modes omit the other category. With no selected entry use component ParameterEffect.'}
                        terms = [source['component']]
                        if not selected: terms.append('ParameterEffect')
                        if task['mode'] != 'fields': terms.extend(e['args'][3] for e in selected)
                        if task['mode'] != 'mechanisms': terms.extend(e['args'][0] for e in selected)
                        expected = {'query': ' '.join(dict.fromkeys(terms)), 'source_ids': [docid]}
                    examples.append(GeneratedExample(example_id=f"{docid}-{task['task_id']}", task_id=task['task_id'], task=family,
                        instruction=task['instruction'], source_group=source['component'], source_artifact_sha256=source['sha256'],
                        source_path=source['path'], source_excerpt=excerpt, split=source['split'], input=inputs, expected=expected,
                        teacher=task['teacher'], generation_round=task['generation_round']))
    receipt = prepare_generated(examples, output=output/'corpus', source_root=source_root)
    receipt.update(instruction_candidates=len(tasks['accepted'])+len(tasks['rejected']), instructions_retained=len(tasks['accepted']),
        unique_used_instructions=len({e.task_id for e in examples}), source_declarations=len(source_propositions),
        task_counts={s: {t: sum(e.split == s and e.task == t for e in examples) for t in ('ONTOLOGY_EXTRACTION', 'RETRIEVAL_QUERY')} for s in SPLITS},
        empty_ontology_counts={s: sum(e.split == s and e.task == 'ONTOLOGY_EXTRACTION' and not e.expected['triples'] for e in examples) for s in SPLITS},
        task_file_sha256=hashlib.sha256(task_file.read_bytes()).hexdigest(),
        generation_method='Session assistant authored two task rounds; AST source binding constructs instances. No external teacher API. No human validation/test examples used.')
    (output/'generation-receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n')
    (output/'task-filter.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(receipt, ensure_ascii=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--tasks', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args(); build(a.source_root.resolve(), a.tasks.resolve(), a.output.resolve())

if __name__ == '__main__': main()
