# AstroGraph R21 — Real-data failure mechanisms

Document status: `CURRENT_R21_RESEARCH_NOTE`.

R21 uses the R20.1 real-data predictions, truth, partitions and results as read-only evidence. It did not rerun inference on DXC09 competition/rematch, OPS-SAT `train=0`, or ESA confirmatory/external partitions. All 79 frozen files remained byte-identical. The resulting status is `DEVELOPMENT_REMEDIATION_READY_FOR_NEW_REPLICATION`; qualification remains `NOT_FLIGHT_QUALIFIED`.

## DXC09 diagnosis

Across 158 scored cases, exact fault-set accuracy remains 0. The frozen output contains 4451 predicted fault elements for 187 true elements, with 62 matches. Micro precision is 0.013929, micro recall is 0.331551, and faulty-case any-overlap is 0.463636.

The decisive pattern is candidate explosion: all 158 cases are over-isolated, while 88 also omit at least one true element. Of 110 faulty cases, 51 have partial overlap and 59 completely miss the true set. This shows that the predictor carries some component/fault information but does not solve exact-set isolation. The component/mode mismatch categories are diagnostic proxies only and do not by themselves prove an ontology defect.

## OPS-SAT raw coverage

The frozen raw track scores 422/529 cases (79.7732%) and blocks 107/529 (20.2268%). Blockage is label-dependent: anomaly 3/113 and nominal 104/416. All 107 blocked cases are in the 5-second sampling stratum. Therefore the covered subset cannot be treated as a random sample of all external cases. Raw and precomputed-feature tracks remain separate.

## ESA confirmatory miss

The frozen confirmatory interval contains 73,641 observable decision points, zero positive alarm points, and four annotated events; TP=0 and FN=4. Hash-bound preprocessing caches and archived fit parameters show nonzero deviations but every channel range remains strictly inside its frozen six-scale threshold interval. This supports `RESPONSE_BELOW_THRESHOLD` at the threshold stage without regenerating score vectors or predictions.

Each confirmatory annotation lasts 3.912–9.912 seconds and contains no timestamp on the five-minute decision grid. This establishes a temporal representation limitation for those annotations. It does not prove that median aggregation erased the raw anomaly. Exact historical score arrays and raw-to-window response were not retained, so stronger causal attribution is withheld.

## Development-only remediation

OPS compared exactly nine preregistered trials on `train=1`. The selected candidate is `LENGTH_AWARE_FALLBACK_FEATURE_SET` at threshold 12, with development coverage 100.0%, F0.5=0.324675, and selective 0/1 risk=0.528634. These are development-selection results, not external performance estimates.

ESA compared exactly four preregistered development candidates. `MEDIAN_5MIN_THRESHOLD_6` was selected with development event TP=3, FP=0, FN=0, and corrected eventwise F0.5=0.999977. The very high development result cannot repair the already-seen confirmatory miss and requires new independent replication.

No new `*_external_validation_receipt.json` was produced. The next valid performance step is a newly reserved holdout in R22 or R23.
