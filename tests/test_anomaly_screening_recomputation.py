from astrograph.retrieval.anomaly_screening_audit import independently_render_anomaly_screening
from astrograph.retrieval.document_corpus_builder import build_causal_reports


def test_independent_anomaly_renderer_matches_registered_production_bytes(tmp_path) -> None:
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text(
        "time_s,x,y,mode\n"
        "0,1,4,A\n10,2,4,A\n20,3,4,A\n30,4,4,A\n"
        "40,8,4,B\n50,9,4,B\n60,10,4,B\n",
        encoding="utf-8",
    )
    reports = build_causal_reports(telemetry, requested_cutoffs_s=(60.0,))
    assert len(reports) == 1
    cutoff, _, production = reports[0]
    independent = independently_render_anomaly_screening(telemetry, cutoff_s=cutoff)
    assert independent == production
    assert len(independent.encode("utf-8")) == 4096
