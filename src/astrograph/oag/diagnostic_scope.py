"""Separate observed functional effect from unresolved component root causes."""

def diagnostic_scope(fault_id: str) -> dict:
    if fault_id == 'ADCS_RW_TORQUE_AUTHORITY_LOSS':
        return {'fault_id':fault_id, 'diagnostic_level':'FUNCTIONAL_EFFECT',
            'effect':'Reduced or absent commanded torque transmission',
            'root_cause_confirmed':False,
            'root_cause_candidates':['MOTOR','DRIVE_ELECTRONICS','POWER_PATH','TORQUE_COMMAND_PATH'],
            'root_cause_candidates_exhaustive':False,
            'required_observations':['Independent applied-torque estimate',
                'Motor phase current and winding continuity',
                'Drive enable, fault registers and command acknowledgement',
                'Actuator power bus voltage and current'],
            'claim_boundary':'Telemetry effect class does not identify an electrical or mechanical root cause.'}
    return {'fault_id':fault_id, 'diagnostic_level':'CONFIGURED_FAULT_HYPOTHESIS',
            'root_cause_confirmed':False,
            'claim_boundary':'Interpret within the evidence package applicability; a supported hypothesis is not flight root-cause qualification.'}
