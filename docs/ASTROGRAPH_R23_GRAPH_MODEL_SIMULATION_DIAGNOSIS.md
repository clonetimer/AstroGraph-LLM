# AstroGraph R23 — Graph / Model-Based / Simulation Diagnosis

Status: **CURRENT_WORKING_PATCH_R23 / PRE-HOLDOUT ENGINEERING REMEDIATION**  
Baseline: immutable `PATCH-R22` unified source; R22 external/seen partitions are not reopened for `FIT/TUNE/SELECT/CALIBRATE/OPTIMIZE`.

## 1. Purpose

R23 begins a core-capability remediation rather than increasing LLM authority. The target architecture is:

```text
admitted engineering catalogs / telemetry
        |
        +--> deterministic system/engineering graph
        |
        +--> admitted observation inconsistencies
                    |
                    v
             model conflicts
                    |
                    v
          bounded minimal diagnoses
                    |
        +-----------+----------------+
        |                            |
  no simulator admitted        governed simulator/trace provider
        |                            |
        v                            v
 model diagnosis only       counterfactual trace comparison
        |                            |
        +-------------+--------------+
                      v
              shadow review result
                      |
              controlled Agent may inspect
```

The Agent remains an investigation planner/orchestrator. It does not create model conflicts, simulator parameters, formal EvidenceFacts, thresholds, or release decisions.

## 2. New deterministic capabilities

### 2.1 DXC command-instance semantics and system-model graph

`src/astrograph/evaluation/nasa_adapt_adapter.py` now preserves explicit XML `component/command` bindings and separates nominal modes from fault modes while retaining the existing catalog schema/version for compatibility.

New graph module:

- `src/astrograph/knowledge/system_model_graph.py`

It constructs content-bound nodes for system, component types, component/sensor instances, commands and fault modes, plus explicit `COMMANDS`, `CONNECTED_TO`, `INSTANCE_OF`, `HAS_FAULT_MODE` and containment edges.

On the supplied public `AdaptDXC.xml`, without reading any scenario truth:

- 173 component instances;
- 172 explicit connections;
- 27 explicit instance command bindings;
- 0 ambiguous command IDs;
- 222 component/fault-mode targets;
- 452 graph nodes;
- 796 graph edges;
- verified example: `EY136_OP -> CB136`.

Machine audit: `governance/evidence/research/r23_graph_sim_diagnosis/PATCH_R23_DXC_SYSTEM_GRAPH_AUDIT.json`.

Important limitation: the public system XML gives structural command binding, but it does not by itself encode the complete hybrid transition/physics semantics needed to infer, for example, that a Boolean command means an exact physical breaker transition. R23 therefore does not invent those semantics from command names.

### 2.2 Bounded conflict / minimal-diagnosis engine

New module:

- `src/astrograph/diagnosis/model_based.py`

The engine computes deterministic hitting sets over admitted conflicts and **keeps subset-minimal diagnoses distinct from minimum-cardinality diagnoses**. Search limits are explicit:

- `max_cardinality`;
- `max_candidates`;
- `max_search_nodes`.

If a bound can exclude diagnoses, the result is `search_complete=false` with reason codes; the omitted candidates are not treated as impossible.

Regression example:

```text
conflict 1 = {A, B}
conflict 2 = {B, C}

subset-minimal diagnoses = {B}, {A,C}
minimum-cardinality tier  = {B}
```

Thus a real two-fault explanation is not discarded merely because a one-fault explanation exists.

R23 currently provides a conservative `structural_conflicts_from_mismatches()` bridge. It uses admitted sensor/component topology and explicit command targets. This is **candidate generation**, not yet a full HyDE-style hybrid physical consistency checker.

### 2.3 Governed simulation/counterfactual interface

New module:

- `src/astrograph/simulation/diagnostic_provider.py`

The zero-extra-dependency R23 provider is `PrecomputedTraceSimulationProvider`. It receives only host-admitted, content-bound traces and does not open arbitrary files or accept an LLM-authored simulator configuration.

Counterfactual comparison requires:

- exact matching time grid;
- explicit comparison channels;
- frozen positive channel scales;
- finite values;
- content-bound trace identities.

It reports normalized RMSE/MAE with semantics:

`DISTANCE_LOWER_IS_BETTER_NOT_PROBABILITY`.

No distance is converted into a posterior fault probability.

Future live Basilisk/MATLAB/Simulink/Sat connectors may implement the same provider contract as optional backends. They are not AstroGraph core dependencies.

### 2.4 Integrated model + simulation diagnosis

New module:

- `src/astrograph/diagnosis/model_simulation.py`

Execution can stop safely after model-based diagnosis when no simulator is admitted, or continue to counterfactual comparison when a governed provider is available.

Synthetic protocol receipt:

- `governance/evidence/research/r23_graph_sim_diagnosis/PATCH_R23_MODEL_SIMULATION_SYNTHETIC_RECEIPT.json`

The synthetic receipt proves executable composition only. It is not diagnostic-accuracy, calibration, external-validation, HIL or flight evidence.

### 2.5 Controlled Agent probes

`DiagnosticProbeKind` adds:

- `MINIMAL_DIAGNOSIS_REVIEW`;
- `SIMULATION_COUNTERFACTUAL_REVIEW`.

Both remain v1 `shadow-only`:

- `formal_evidence_eligible=false`;
- no EvidenceFact generation;
- no diagnosis/release mutation.

They are **not** automatically exposed by the ordinary EvidenceBundle margin catalog. A trusted host must first construct a content-bound model-conflict or simulation-trace catalog. The LLM can then select only the frozen `probe_id`.

## 3. Engineering graph construction from the newly supplied Sat package

New module:

- `src/astrograph/knowledge/engineering_asset_graph.py`

The builder consumes explicit engineering registries and admits only exact unique entity links. It does not silently fuzzy-match component names.

Using the newly supplied `Sat.zip`:

- ComponentRegistry: 19 components;
- FaultCatalog: 61 fault records;
- SignalRegistry: 106 signals;
- constructed graph: 186 nodes / 133 edges;
- exact fault-to-component bindings: 54/61 fault nodes;
- exact signal-group-to-component bindings: 67/106 signal nodes;
- unresolved fault component labels include `光纤陀螺` and `蓄电池`;
- unresolved telemetry groups remain explicit rather than being auto-linked.

The `IDENTIFIER_HIERARCHY_PARENT` edge means only a dotted identifier naming hierarchy. It is not promoted to a physical `PART_OF` or causal relation.

Machine graph: `governance/evidence/research/r23_graph_sim_diagnosis/PATCH_R23_ENGINEERING_ASSET_GRAPH.json`.

## 4. New data-asset admission boundary

R23 inventories the newly supplied archives without consuming embedded evaluation outcomes for model selection:

- ADCS v5.2.2 archive: role `DATASET_GOVERNED_SCHEMA_ONLY_NO_R23_MODEL_SELECTION`;
- PHM v6.0 archive: same role;
- Sat simulation platform archive: role `OPTIONAL_SIMULATION_BACKEND_SOURCE`.

Machine receipt:

- `governance/evidence/research/r23_graph_sim_diagnosis/PATCH_R23_ASSET_INVENTORY.json`.

The ADCS/PHM archives contain dataset/evaluation material. R23 only reads archive identity and allow-listed schema-level manifest facts in this patch. Per-case labels, embedded predictions and performance reports are not reused for R23 fitting/tuning/selection/calibration/optimization.

## 5. Relation to public algorithms

R23's direction is consistent with established model-based diagnosis rather than treating an LLM as the diagnosis engine:

1. NASA HyDE describes hybrid model-based diagnosis that maintains candidates, predicts expected system behavior, checks observations for consistency and uses inconsistencies to generate successor candidates.  
   NASA NTRS: https://ntrs.nasa.gov/citations/20120007370

2. Conflict-directed model-based diagnosis uses the duality between conflicts and diagnoses; diagnoses are hitting sets of conflicts. R23 implements the bounded deterministic hitting-set layer, while the stronger hybrid consistency checker remains future work.  
   Stern et al., AAAI: https://doi.org/10.1609/aaai.v26i1.8231

3. Hitting-set literature distinguishes enumeration/search properties from physical consistency checking. The expensive domain-specific step is often the consistency/conflict computation, which is why R23 does not present the new hitting-set code as a complete DXC diagnosis model.  
   Jannach et al., AAAI: https://doi.org/10.1609/aaai.v29i1.9389

4. Basilisk is an open spacecraft-centric simulation framework supporting faster-than-real-time, Monte-Carlo and real-time/HIL workflows. It is an appropriate future optional provider, but R23 does not add it as a base dependency.  
   Basilisk documentation: https://avslab.github.io/basilisk/

The separately supplied Sat package contains MATLAB/Simulink model/verification/FDIR assets and is likewise treated as an optional simulation backend source, not a core AstroGraph runtime dependency.

## 6. What R23 does not claim

R23 does **not** claim that:

- current DXC performance has improved on R20/R21 seen external partitions;
- structural graph neighborhoods are equivalent to a complete hybrid component model;
- minimum-cardinality diagnoses are the only valid diagnoses;
- simulation distance is a probability;
- new ADCS/PHM embedded evaluation results were used for development selection;
- OPS/ESA frozen R22 algorithm identity has changed;
- shadow probes are formal evidence producers;
- MATLAB/Simulink or Basilisk automatic execution is already installed/configured;
- the implementation is HIL/target/flight qualified.

## 7. Next engineering/research steps

1. Add a true component behavior/transition contract for ADAPT commandable components and compute conflicts by prediction-vs-observation consistency, replacing structural-neighborhood conflicts where models are available.
2. Add a governed live simulation provider adapter, first for a non-core optional backend, with exact model/config/input/output receipts and deterministic post-run identity checks.
3. Construct an explicit, human-reviewed alias/mapping layer for the unresolved Sat component/signal registry links rather than using fuzzy auto-linking.
4. Keep OPS feature-model remediation and ESA high-resolution temporal remediation as separate pre-holdout workstreams; do not mix those algorithm changes into this graph/simulation patch.
5. Evaluate diagnostic effect only under a newly frozen development/holdout protocol that preserves the R21/R22 exposure gates.
