from astrograph.evaluation.r30_simulation_blind_campaign import (
    generate_blind_cases,predict_cases_truth_free,evaluate_frozen_predictions,
)


def test_prediction_does_not_consume_hidden_truth():
    cases=generate_blind_cases(3001)[:3]
    pred1=predict_cases_truth_free(cases)
    changed=[]
    for c in cases:
        payload=c.model_dump(mode='json')
        payload['hidden_fault_ids']=('LGT_C:FailedOff',) if c.hidden_fault_ids != ('LGT_C:FailedOff',) else ('LGT_A:FailedOff',)
        payload.pop('case_sha256')
        import hashlib,json
        raw=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        payload['case_sha256']=hashlib.sha256(raw).hexdigest()
        changed.append(type(c).model_validate(payload))
    pred2=predict_cases_truth_free(changed)
    assert [x.predicted_fault_ids for x in pred1]==[x.predicted_fault_ids for x in pred2]


def test_blind_campaign_freezes_predictions_and_reports_model_mismatch_performance():
    cases=generate_blind_cases(3001)
    pred=predict_cases_truth_free(cases)
    report=evaluate_frozen_predictions(cases=cases,predictions=pred,seed=3001)
    assert report.case_count==48
    assert 0.5 <= report.exact_set_accuracy < 1.0
    assert 0.5 <= report.fault_element_f1 < 1.0
    assert report.status=='SYNTHETIC_BLIND_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION'
