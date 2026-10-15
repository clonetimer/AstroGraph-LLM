# AstroGraph Research Backlog after R18

Status: CURRENT_R18_RESEARCH_BACKLOG

R18 freezes group-aware calibration-transfer, fixed-threshold transfer, and BiRRF leave-branch-out protocols. Real external validation remains the dominant research gap.

| ID | Priority | Status | Remaining work |
|---|---|---|---|
| R18-CAL-REAL | P0-RESEARCH | OPEN | Run the frozen calibration-transfer design on real frozen independent groups; report NLL/Brier/ECE/AURC and classwise metrics with group uncertainty. |
| R18-CAL-FIT | P0-RESEARCH | OPEN | Preregister the actual calibrator family/fit objective and fit only on `CALIBRATION_FIT`; compare raw vs calibrated predictions without selecting on confirmatory groups. |
| R18-OOD-RISK | P0-RESEARCH | OPEN | Freeze corruption/OOD sets and transfer a development-selected abstention threshold without retuning; report risk/coverage by shift stratum. |
| R18-FUS-REAL | P0-RESEARCH | OPEN | Execute R17 weight/smoothing grid plus R18 leave-branch-out on preregistered real independent groups and compare against data-only/KG-only/simple baselines. |
| R18-SHIFT-DESIGN | P0-RESEARCH | OPEN | Add independently varied simulator/mode/severity/noise groups; current jointly shifted scenarios cannot identify single-axis effects. |
| R18-SAMPLE-SIZE | P1-RESEARCH | OPEN | Expand independent trajectory groups and preregister precision/sample-size targets for calibration-transfer intervals and group-direction tests. |
| R18-ECE-ROBUSTNESS | P1-RESEARCH | OPEN | Compare frozen ECE estimators/bin counts or a calibration test on development data; do not choose the estimator from confirmatory results. |
| R18-CAL-RECEIPT-V2 | P1-GOVERNANCE | OPEN | Consider a future calibration receipt binding NLL/Brier/classwise/transfer report identities, not only validation ECE, after a real campaign exists. |
| R18-KG-PROVENANCE | P1-RESEARCH | OPEN | Extend source-dependence audit to leave-edge/source-out by provenance grade, age, contradiction status and causal-vs-associational edge type. |
| R14-TMP-001 | P1-RESEARCH | OPEN | Add false-alarm/ARL, event-level detection/isolation delay and cadence/window sensitivity. |
| R14-ADM-001 | P1-GOVERNANCE | OPEN | Bind sensor/calibration version, acquisition provenance, cadence/gaps/coverage and measurement uncertainty. |

Highest-value next execution is not another Agent feature. It is a real frozen external validation package with genuinely independent groups and deliberately separated shift axes.
