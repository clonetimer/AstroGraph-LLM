"""ESA event semantics. Evaluation remains gated on OPS-SAT and DXC admission."""
SEMANTICS={'Anomaly':'ANOMALY','Rare Event':'RARE_NOMINAL','Communication Gap':'COMMUNICATION_GAP'}

def event_category(category):
    if category not in SEMANTICS:raise ValueError('ESA_EVENT_SEMANTICS_UNADMITTED')
    return SEMANTICS[category]

def require_fault_evidence(category):
    # An ESA annotation establishes an event, never a root-cause fault.
    raise ValueError('ANOMALY_EVENT_NOT_FORMAL_FAULT_EVIDENCE')

def require_evaluation_prerequisites(*, ops_sat_passed, dxc_passed):
    if not ops_sat_passed or not dxc_passed:raise ValueError('ESA_EVALUATION_BLOCKED_PREREQUISITES')
