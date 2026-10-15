"""Explicit scenario annotations, including nominal N/A and continuation faults.

Workbook names are used for exact joins only, never to infer labels or as model
inputs. xlrd is an offline evaluator dependency, not an online predictor input.
"""
from __future__ import annotations
from pathlib import PurePosixPath
import math

def parse_annotation_rows(rows):
    cases={};current=None;header=False
    for row in rows:
        if len(row)<4:raise ValueError('WORKBOOK_SCHEMA_UNADMITTED')
        filename,component,mode,time=row[:4]
        # These supplied workbooks explicitly end the case table with NOTES or
        # a numeric nominal/multiple/single aggregate. Never skip case errors.
        if header and (str(filename).strip().upper().startswith('NOTES:') or
          (isinstance(filename,(int,float)) and str(mode).strip()=='nominal')):
            break
        if filename=='File Name':
            if list(row[1:4])!=['Component','Mode','Time (sec)']:raise ValueError('WORKBOOK_SCHEMA_UNADMITTED')
            header=True;continue
        if not header:continue
        if not any(str(v).strip() for v in row):continue
        if filename:
            name=PurePosixPath(str(filename).strip()).name
            current=name[:-4] if name.lower().endswith(('.txt','.scn','.mat')) else name
            if current in cases:raise ValueError('DUPLICATE_SCENARIO_ANNOTATION')
            cases[current]={'true_fault_ids':[],'fault_events':[],'explicit_nominal':False}
        if current is None:raise ValueError('FAULT_CONTINUATION_WITHOUT_CASE')
        c=cases[current];component=str(component).strip();mode=str(mode).strip()
        if component=='N/A':
            if mode or time not in ('',None) or c['true_fault_ids']:raise ValueError('CONTRADICTORY_NOMINAL_ANNOTATION')
            c['explicit_nominal']=True;continue
        if not component or not mode or c['explicit_nominal']:raise ValueError('FAULT_ANNOTATION_INCOMPLETE')
        t=float(time)
        if not math.isfinite(t) or t<0:raise ValueError('INJECTION_TIME_INVALID')
        fault=component+':'+mode
        if fault in c['true_fault_ids']:raise ValueError('DUPLICATE_FAULT_ANNOTATION')
        c['true_fault_ids'].append(fault);c['fault_events'].append({'fault_id':fault,'injection_time_s':t})
    if not cases:raise ValueError('WORKBOOK_NO_ADMITTED_CASES')
    for c in cases.values():
        c['true_fault_ids'].sort();c['schema_version']='astrograph-fault-set-truth.v1'
        c['status']='OFFICIAL_WORKBOOK_ANNOTATION_PARSED'
    return cases

def read_workbook_bytes(content):
    import xlrd
    workbook=xlrd.open_workbook(file_contents=content)
    result={}
    for sheet in workbook.sheets():
        parsed=parse_annotation_rows([sheet.row_values(i) for i in range(sheet.nrows)])
        if result.keys()&parsed.keys():raise ValueError('DUPLICATE_SCENARIO_ANNOTATION')
        result.update(parsed)
    return result

def verify_injection_agreement(annotation, scn_truth, *, time_tolerance_s=.001):
    if annotation['true_fault_ids']!=scn_truth['true_fault_ids']:raise ValueError('WORKBOOK_SCN_FAULT_SET_MISMATCH')
    scn={f:r['time_tick']/1000. for r in scn_truth['injection_events'] if r['is_injection'] for f in r['fault_ids']}
    for e in annotation['fault_events']:
        if abs(e['injection_time_s']-scn[e['fault_id']])>time_tolerance_s:raise ValueError('WORKBOOK_SCN_INJECTION_TIME_MISMATCH')
    return {'tick_to_second_scale':.001,'basis':'matched explicit Time (sec) annotations to all SCN injection records','verified_fault_event_count':len(scn)}
