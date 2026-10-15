# CURRENT_ROADMAP — AstroGraph after R32

## Active phase: AstroGraph-Bench/UI

Core feature expansion is frozen for the current spacecraft engineering scope. Active work moves to benchmark operations, operator UX, evidence presentation and independent validation support.

### Bench/UI P0

1. Unified benchmark job model for ADCS, PHM, OPS-SAT, ESA and DXC/ADAPT.
2. Dataset/profile admission preflight with explicit data-exposure state.
3. Metric-contract-aware comparison tables; never compare incompatible metrics as one leaderboard.
4. Qualification badges: development, seen-external audit, new-holdout external validation, synthetic protocol, HIL/flight.
5. Diagnostic drill-down: detection → localization → classification → mechanism/simulation evidence.
6. DXC conflict/minimal-diagnosis/expected-response/simulation inspector.
7. ESA fast/slow branch timeline and event grouping visualization.
8. Agent tool inspector showing host-bound parameters and zero release authority.
9. Reproducibility/receipt download and source/model/dataset identity display.
10. Stable vs experimental algorithm selection must be explicit and governed.

## Validation work that may change qualification but should not expand core features by itself

- genuinely new ADCS trajectory groups;
- unknown-onset PHM evaluation;
- genuinely new OPS-like holdout;
- genuinely new ESA Mission1-like high-resolution holdout;
- genuinely new ADAPT/DXC holdout;
- HIL/target-hardware campaigns.

If such campaigns expose a concrete missing spacecraft function, reopen the core under a new scoped remediation patch. Otherwise treat them as qualification work.

## Roadmap-only feature expansion

The following are no longer active R32 core scope:

- additional numerical mechanism families added only for breadth;
- non-spacecraft datasets and domain-specific features derived from them;
- automatic high-fidelity Simulink/Basilisk/VirtualADAPT providers beyond the current provider contract;
- formal literature-voting promotion;
- autonomous control authority;
- flight qualification.
