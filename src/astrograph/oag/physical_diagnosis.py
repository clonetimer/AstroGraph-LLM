"""Causal paired-observation diagnosis for the registered RW simulation scope.

Physical transmission and sensor-consistency tests are a parallel diagnostic
channel, not probabilities derived from a detector's class mask. No labels or
fault-onset metadata are accepted. Fixed horizons avoid searching a truth onset.
"""
from __future__ import annotations
import hashlib
import numpy as np
from .diagnostic_scope import diagnostic_scope

RW_FIELDS = [f'{stem}_{j}' for j in range(4) for stem in (
    'adcs.rw.command_torque_nm', 'adcs.control.applied_torque_nm', 'adcs.rw.measured_speed_rad_s')]


def physical_signatures(source, nominal, *, horizons=(20,8,4)):
    if tuple(horizons) != (20,8,4):
        raise ValueError('unqualified horizon policy')
    if len(source)<20:
        raise ValueError('twenty observations required')
    s=source.iloc[-20:].copy()
    n=nominal.loc[nominal.time_s.between(s.time_s.iloc[0],s.time_s.iloc[-1])].copy()
    times=s.time_s.to_numpy(float)
    if len(s)!=20 or not np.array_equal(times,n.time_s.to_numpy(float)):
        raise ValueError('twenty aligned scheduled observations are required')
    if not np.isfinite(times).all() or not np.all(np.diff(times)>0):
        raise ValueError('ordered finite observation times required')
    cadence=float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times),cadence,rtol=1e-6,atol=1e-9):
        raise ValueError('irregular scheduled grid is not qualified')
    a,b=s[RW_FIELDS].to_numpy(float),n[RW_FIELDS].to_numpy(float)
    valid=np.isfinite(a).all(axis=1)&np.isfinite(b).all(axis=1)
    hashed_fields=['time_s',*RW_FIELDS,*[f'adcs.sensor.gyro_measured_rad_s_{x}' for x in 'xyz'],
        *[f'adcs.innovation.star_gyro_rate_residual_rad_s_{x}' for x in 'xyz'],
        'adcs.star_tracker.valid','adcs.star_tracker.solution_fresh']
    hashed_fields=[c for c in hashed_fields if c in s and c in n]
    receipt={'source_sha256':hashlib.sha256(s[hashed_fields].to_csv(index=False).encode()).hexdigest(),
        'nominal_sha256':hashlib.sha256(n[hashed_fields].to_csv(index=False).encode()).hexdigest(),
        'hashed_fields':hashed_fields,
        'window_start_s':float(times[0]),'window_end_s':float(times[-1]),
        'valid_pairs':int(valid.sum()),'scheduled_pairs':20,'missing_times_s':times[~valid].tolist(),
        'imputation_performed':False,'model_required':False}
    latest=float(times[np.flatnonzero(valid)[-1]]) if valid.any() else None
    receipt['latest_observation_time_s']=latest
    receipt['observation_age_s']=float(times[-1]-latest) if latest is not None else None
    if valid.sum()<18 or latest is None or times[-1]-latest>cadence+1e-9:
        return {'status':'INSUFFICIENT_OBSERVATIONS','hypotheses':[],'receipt':receipt}
    def transfer(cmd,applied):
        rms=float(np.sqrt(np.mean(cmd**2)))
        gain=float(np.dot(cmd,applied)/max(np.dot(cmd,cmd),1e-20))
        error=float(np.sqrt(np.mean((applied-gain*cmd)**2))/max(rms,1e-12))
        return rms,gain,error
    # A speed offset can also be accumulated physical motion after an earlier
    # actuator fault. Current-window command agreement alone is insufficient.
    history=source.loc[source.time_s<=times[-1]]
    hn=nominal.loc[nominal.time_s.between(history.time_s.iloc[0],times[-1])]
    history_agrees=False
    if np.array_equal(history.time_s.to_numpy(),hn.time_s.to_numpy()):
        ha,hb=history[RW_FIELDS].to_numpy(float),hn[RW_FIELDS].to_numpy(float)
        good=np.isfinite(ha).all(axis=1)&np.isfinite(hb).all(axis=1)
        command_indices=[k for k in range(12) if k%3!=2]
        if good.any():
            first=np.flatnonzero(good)[0]
            history_agrees=bool(history.time_s.iloc[first]<=1.0 and
                np.max(abs(ha[first,2::3]-hb[first,2::3]))<=.2 and
                np.max(abs(ha[good][:,command_indices]-hb[good][:,command_indices]))<=2e-5)
    receipt['command_history_matches_reference']=history_agrees
    receipt['history_source_sha256']=hashlib.sha256(history[['time_s',*RW_FIELDS]].to_csv(index=False).encode()).hexdigest()
    receipt['history_nominal_sha256']=hashlib.sha256(hn[['time_s',*RW_FIELDS]].to_csv(index=False).encode()).hexdigest()
    # Assumed applied-torque error bound for this simulation experiment; not a
    # calibrated flight sensor specification. A gain interval limits low-input
    # decisions instead of merely lowering the old fixed excitation threshold.
    torque_error_bound=1e-6
    receipt['assumed_torque_error_bound_nm']=torque_error_bound
    hypotheses={}
    for width in horizons:
        indices=np.flatnonzero(valid & (np.arange(20)>=20-width))
        if len(indices)<4:continue
        for j in range(4):
            cmd,applied,speed=a[indices,j*3:j*3+3].T
            nc,na,ns=b[indices,j*3:j*3+3].T
            rms,gain,error=transfer(cmd,applied);nrms,ngain,nerror=transfer(nc,na)
            delta=speed-ns
            values={'command_rms':rms,'gain':gain,'fit_error':error,'nominal_command_rms':nrms,
                'nominal_gain':ngain,'nominal_fit_error':nerror,'speed_absmax':float(np.max(abs(speed))),
                'nominal_speed_absmin':float(np.min(abs(ns))),'offset_absmean':float(np.mean(abs(delta))),
                'offset_std':float(np.std(delta)),'paired_command_absmax':float(np.max(abs(cmd-nc)))}
            radius=torque_error_bound*float(np.sum(abs(cmd)))/max(float(np.dot(cmd,cmd)),1e-20)
            nominal_radius=torque_error_bound*float(np.sum(abs(nc)))/max(float(np.dot(nc,nc)),1e-20)
            values['gain_error_radius']=radius
            reference_ok=rms>=1e-5 and nrms>=1e-5 and radius<=.1 and nominal_radius<=.1 and .9<=ngain<=1.1 and nerror<=.1
            if not reference_ok:continue
            matches=[]
            if np.max(abs(speed))<=1. and np.min(abs(ns))>=2. and abs(gain-1)>=.15:
                matches.append(('ADCS_RW_JAM','LOCKED_WHEEL_WITH_TORQUE_MISMATCH'))
            # Stable transmission slope rejects mixed healthy/fault segments.
            if gain>=-radius and gain+radius<=.85 and error<=max(.01,2*torque_error_bound/max(rms,1e-12)) and np.max(abs(speed))>=2.:
                matches.append(('ADCS_RW_TORQUE_AUTHORITY_LOSS','STABLE_ATTENUATED_TORQUE_TRANSMISSION'))
            if history_agrees and np.mean(abs(delta))>=5. and np.std(delta)<=.2 and abs(gain-1)<=.05 and error<=.01 and np.max(abs(cmd-nc))<=2e-5:
                matches.append(('ADCS_RW_SPEED_SENSOR_BIAS','OFFSET_WITH_UNCHANGED_COMMAND_AND_TRANSMISSION'))
            # An imposed braking torque opposes measured rotation, differs
            # from the command, and removes speed. A proportional partial loss
            # is explicitly excluded; it is not evidence of a jam transient.
            brake_power=applied*speed
            if (np.min(abs(ns))>=2. and abs(speed[0])-abs(speed[-1])>=2.
                    and np.all(np.diff(abs(speed))<=.02)
                    and np.all(brake_power<=1e-8) and np.count_nonzero(brake_power < -1e-5)>=len(speed)-1
                    and (gain < -radius or gain>1.1+radius or error>.1)):
                matches.append(('ADCS_RW_JAM','UNCOMMANDED_BRAKING_AND_SPEED_DECAY'))
            for fid,signature in matches:
                # Retain the longest supporting horizon for each wheel/fault.
                hypotheses.setdefault((fid,j),{'fault_id':fid,'wheel_index':j,'signature':signature,
                    'scheduled_horizon':width,'supporting_times_s':times[indices].tolist(),
                    'observed_pairs':len(indices),'values':values})
    # The recorder innovation compares a quaternion interval rate with the
    # endpoint gyro. Use the trapezoidal gyro rate on the same interval before
    # comparing with the paired reference; otherwise acceleration looks like bias.
    gyro=[f'adcs.sensor.gyro_measured_rad_s_{axis}' for axis in 'xyz']
    innovation=[f'adcs.innovation.star_gyro_rate_residual_rad_s_{axis}' for axis in 'xyz']
    quality=['adcs.star_tracker.valid','adcs.star_tracker.solution_fresh']
    # A gyro-only explanation is not isolated while actuator response has an
    # independent mismatch, even when a wheel is still in its braking transient.
    actuator_mismatch=False
    for j in range(4):
        mismatch=abs((a[valid,j*3]-a[valid,j*3+1])-(b[valid,j*3]-b[valid,j*3+1]))
        actuator_mismatch |= np.count_nonzero(mismatch>np.maximum(2e-6,.05*abs(a[valid,j*3])))>=4
    receipt['independent_actuator_mismatch']=bool(actuator_mismatch)
    if not actuator_mismatch and all(c in s and c in n for c in gyro+innovation+quality):
        g,ng=s[gyro].to_numpy(float),n[gyro].to_numpy(float)
        residual,nresidual=s[innovation].to_numpy(float),n[innovation].to_numpy(float)
        aligned=residual[1:]+.5*np.diff(g,axis=0)
        naligned=nresidual[1:]+.5*np.diff(ng,axis=0)
        delta=aligned-naligned
        good=np.isfinite(delta).all(axis=1)
        q=(s[quality].to_numpy(float)==1).all(axis=1)&(n[quality].to_numpy(float)==1).all(axis=1)
        good &= q[1:]&q[:-1]
        for width in (20,8,4):
            indices=np.flatnonzero(good & (np.arange(19)>=20-width))
            if len(indices)<4:continue
            values=delta[indices];mean=values.mean(axis=0)
            amplitude=float(np.linalg.norm(mean));spread=float(np.sqrt(np.mean(np.sum((values-mean)**2,axis=1))))
            if amplitude>=4e-5 and spread<=.25*amplitude:
                hypotheses.setdefault(('ADCS_GYRO_BIAS_DRIFT',None),{
                    'fault_id':'ADCS_GYRO_BIAS_DRIFT','wheel_index':None,
                    'signature':'PAIRED_TIME_ALIGNED_GYRO_BIAS','scheduled_horizon':width,
                    'supporting_times_s':times[indices+1].tolist(),'observed_pairs':len(indices),
                    'values':{'bias_vector_rad_s':mean.tolist(),'bias_norm_rad_s':amplitude,'temporal_spread_rad_s':spread}})
    return {'status':'EVALUATED','hypotheses':list(hypotheses.values()),'receipt':receipt}


def diagnose_physical_complement(source, nominal, baseline_summary=None):
    """Keep broad model diagnosis as fallback; use specific physical evidence.

    Gyro residual envelopes cannot veto an independently consistent RW signature.
    Multiple fault types or wheels remain candidates; no single-fault truth is
    read or presumed to turn conflicting observations into a definite answer.
    """
    from astrograph.integrations.brb_diagnosis_trial import base_prediction
    physical=physical_signatures(source,nominal)
    if baseline_summary:
        from .contracts import content_hash
        if baseline_summary.get('summary_sha256') != content_hash({k:v for k,v in baseline_summary.items() if k!='summary_sha256'}):
            raise ValueError('baseline summary hash mismatch')
        window=baseline_summary['window'];receipt=physical['receipt']
        if window != {'start_s':receipt['window_start_s'],'end_s':receipt['window_end_s']}:
            raise ValueError('baseline and physical evidence windows differ')
    old=base_prediction(baseline_summary) if baseline_summary else {'kind':'UNAVAILABLE','ids':[]}
    pairs={(h['fault_id'],h['wheel_index']) for h in physical['hypotheses']}
    ids=sorted({p[0] for p in pairs})
    blocked=baseline_summary.get('diagnostic_decision',{}).get('blocking_conflicts',[]) if baseline_summary else []
    if blocked:
        prediction=old;route='PRESERVED_ASSURANCE_BLOCK';location=None
    elif len(pairs)==1:
        fid,wheel=next(iter(pairs));prediction={'kind':'CLASS','ids':[fid]}
        route='SPECIFIC_PHYSICAL_SIGNATURE';location=({'component':'GYROSCOPE'} if wheel is None else {'component':'REACTION_WHEEL','wheel_index':wheel})
    elif pairs:
        prediction={'kind':'SET','ids':ids} if len(ids)<=3 else {'kind':'ABSTAIN','ids':[]}
        route='CONFLICTING_PHYSICAL_SIGNATURES';location=None
    else:
        prediction=old;route='EXISTING_DIAGNOSIS';location=None
    return {'schema_version':'astrograph-physical-complement.v1','prediction':prediction,
        'decision_route':route,'localization':location,'baseline_prediction':old,
        'physical_evidence':physical,'diagnostic_scopes':[diagnostic_scope(fid) for fid in prediction['ids']],
        'probability_calibration':'NOT_ESTABLISHED','labels_used':False,
        'claim_boundary':'Registered paired RW observations and excitation required. Functional torque loss is not motor root-cause identification. No flight or arbitrary-domain qualification.'}
