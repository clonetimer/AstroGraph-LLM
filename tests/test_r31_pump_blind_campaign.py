from astrograph.evaluation.r31_pump_blind_campaign import (
    evaluate_frozen_predictions,
    generate_blind_cases,
    predict_cases_truth_free,
    run_blind_campaign,
)


def test_r31_pump_blind_predictions_freeze_before_truth_join_and_are_reproducible():
    cases = generate_blind_cases(seed=3101, repetitions_per_mode=4)
    predictions = predict_cases_truth_free(cases)
    report = evaluate_frozen_predictions(cases=cases, predictions=predictions, seed=3101)
    report2, predictions2, cases2 = run_blind_campaign(seed=3101, repetitions_per_mode=4)
    assert predictions == predictions2
    assert cases == cases2
    assert report == report2
    assert report.case_count == 8
    assert report.status == "SYNTHETIC_BLIND_EVENT_CONDITIONAL_PROTOCOL_EVIDENCE_NOT_EXTERNAL_VALIDATION"


def test_r31_blind_predictor_source_does_not_consume_hidden_fault_label():
    from pathlib import Path
    text = (Path(__file__).resolve().parents[1] / "src/astrograph/evaluation/r31_pump_blind_campaign.py").read_text(encoding="utf-8")
    function = text.split("def predict_cases_truth_free", 1)[1].split("def evaluate_frozen_predictions", 1)[0]
    assert "hidden_fault_id" not in function
