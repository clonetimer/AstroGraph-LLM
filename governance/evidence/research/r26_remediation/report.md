# PATCH-R26 DXC Discrete-State Consistency + ESA Frozen High-Resolution Detector

## Result

R26 keeps all R20/R21 external negatives immutable.  It changes two development/future-replication capabilities only.

### DXC09 Sample development

R25's numeric residual branch excludes bool `PositionSensor` observations because the continuous path requires verified engineering units and rejects booleans.  The ADAPT engineering XML, however, binds each `PositionSensor` directly to one Relay/CircuitBreaker instance.  R26 therefore adds a separate discrete-state contract rather than weakening the continuous-unit gate.

The selected fusion policy is `DISCRETE_FIRST_SINGLE_CONTINUOUS`: exact uncommanded state-consistency faults have higher engineering authority; if no discrete inconsistency exists, only the top continuous residual hypothesis is emitted to limit over-isolation.

Selection (11 cases): exact set 0.272727, fault-element micro F1 0.400000, component micro F1 0.400000.

Already-seen internal replay (14 cases; development audit only):

- fault-element precision / recall / F1: 0.500000 / 0.437500 / **0.466667**;
- component precision / recall / F1: 0.571429 / 0.500000 / **0.533333**;
- exact fault-set rate: **0.428571**;
- any fault-element overlap: **0.500000**;
- post-injection first-alarm rate: **0.928571**;
- median post-injection delay: **0.328 s**;
- mean predicted fault count: **1.0**.

The same R25 continuous branch on those 14 cases had fault F1 0.095238, component F1 0.190476, exact set 0, overlap 0.142857, and mean predicted fault count 1.857143.

This is a large development improvement, but the replay was already exposed in R25 and cannot be promoted to independent validation.  Position-sensor `Stuck` faults can remain ambiguous with actual adjacent component state faults unless downstream physical-response equations are available.

### ESA

R26 freezes the R25 development-selected raw-grid detector without retuning:

- robust lag-one delta z threshold = **8**;
- minimum coincident channels = **2**;
- alarm grouping gap = **40 s**;
- frozen development reference SHA256 = `7f3d73247e3ec2bed5c1b0461026785fa87a4a40405c47a81f890e0284b6553e`.

A content-bound frozen detector artifact and truth-free prediction receipt path are implemented.  It can be bound to the existing R22 first-contact new-holdout identity chain before labels are joined.

No genuinely new ESA holdout was supplied in R26.  Therefore no new ESA external-performance result exists and the frozen R20/R21 73,641-point / 0-alarm / 4-FN confirmatory result remains unchanged.

## Claim boundary

R26 is development remediation plus a frozen future replication candidate.  It does not create formal EvidenceFact, calibrated probabilities, new DXC Competition/Rematch results, new ESA confirmatory results, external validation, HIL or flight qualification.
