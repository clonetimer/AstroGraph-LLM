# AstroGraph Research Backlog after R17

Status: CURRENT_R17_RESEARCH_BACKLOG

R17 closes the score-semantics ambiguity for the built-in engineering screening path and introduces content-bound calibration receipts plus BiRRF sensitivity tooling. The following items remain open and must not be inferred from the R17 implementation.

| ID | Priority | Status | Remaining work |
|---|---|---|---|
| R17-CAL-TRANSFER | P0-RESEARCH | OPEN | Evaluate calibration transfer across simulator versions, mission modes, fault severity, noise/missingness and independent groups; report NLL/Brier/ECE plus uncertainty intervals. |
| R17-FUS-REAL | P0-RESEARCH | OPEN | Run the frozen BiRRF weight/smoothing grid on preregistered independent-group data and compare against simpler data-only/KG-only/score-fusion baselines. |
| R17-OOD-RISK | P0-RESEARCH | OPEN | Freeze OOD/corruption sets and evaluate risk-coverage/selective risk under distribution shift; confidence thresholds must not be selected on the held-out confirmatory set. |
| R17-KG-DEPENDENCE | P1-RESEARCH | OPEN | Quantify branch/source dependence, leave-edge/source-out utility, edge provenance grades, contradiction/age handling and causal-vs-associational semantics. |
| R17-CAL-RECEIPTS | P1-GOVERNANCE | PARTIAL | Migrate legacy calibrated model packages from ID-only calibration metadata to actual content-bound calibration receipts and verify them at provider admission. |
| R17-SCORE-SCHEMA | P1-ENGINEERING | PARTIAL | `ClassProbability.probability` remains a legacy field name even when `class_value_semantics=NORMALIZED_RANKING_MASS`; a future schema-v2 migration may rename it without breaking stored artifacts. |
| R14-TMP-001 | P1-RESEARCH | OPEN | False-alarm/ARL, detection-delay and cadence/window sensitivity calibration. |
| R14-ADM-001 | P1-GOVERNANCE | OPEN | Bind sensor/calibration version, acquisition provenance, cadence/gaps/coverage and measurement uncertainty. |
| R14-MET-001 | P1-RESEARCH | OPEN | Add event-level false-alarm rate, detection/isolation delay and mission-phase metrics beyond classification summaries. |
| R14-SAM-001 | P1-RESEARCH | OPEN | Increase independent trajectory groups rather than inflating case count within shared groups. |

The highest-value next step is not another UI/Agent feature. It is a preregistered external validation package combining calibration transfer, OOD/selective-risk and BiRRF sensitivity on truly independent groups.
