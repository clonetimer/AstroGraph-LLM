#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from astrograph.evaluation.r30_simulation_blind_campaign import generate_blind_cases,predict_cases_truth_free,evaluate_frozen_predictions

p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=3001);a=p.parse_args()
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
cases=generate_blind_cases(a.seed)
pred=predict_cases_truth_free(cases)
# Freeze truth-free predictions first.
pred_doc={'schema_version':'astrograph-r30-blind-predictions.v1','seed':a.seed,'predictions':[x.model_dump(mode='json') for x in pred]}
(out/'predictions_before_truth_join.json').write_text(json.dumps(pred_doc,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
report=evaluate_frozen_predictions(cases=cases,predictions=pred,seed=a.seed)
truth_doc={'schema_version':'astrograph-r30-blind-truth.v1','seed':a.seed,'cases':[{'case_id':x.case_id,'hidden_fault_ids':x.hidden_fault_ids,'case_sha256':x.case_sha256} for x in cases]}
(out/'truth_after_prediction_freeze.json').write_text(json.dumps(truth_doc,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
(out/'report.json').write_text(json.dumps(report.model_dump(mode='json'),ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf-8')
print(json.dumps(report.model_dump(mode='json'),ensure_ascii=False,sort_keys=True))
