# PHM 成对可观测性定义

角色：SIMULATION_KNOWLEDGE_NOT_CURRENT_OBSERVATION。当前源码核对，非人工批准的诊断真值。

原生成器使用同时间点正常/故障轨迹、量化分辨率和注入前噪声计算首次差异；这是仿真数据内的观测判据，不能称为真实设备故障报告。

来源：scripts/generate_whole_spacecraft_phm_dataset_v6.py:636-717；SHA256：956f07dabfaf600729aa61385dc1901504d33e137912288020f3253538020252。

```python
def _manifestation(
    nominal_rows: Sequence[Mapping[str, Any]],
    anomaly_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Measure the first paired observable divergence after injection."""
    event = float(metadata["event_start_s"])
    activation = float(metadata["activation_time_s"])
    subsystem = str(metadata["subsystem"])
    nominal_by_time = {float(row["time_s"]): row for row in nominal_rows}
    anomaly_by_time = {float(row["time_s"]): row for row in anomaly_rows}
    common_times = sorted(set(nominal_by_time) & set(anomaly_by_time))
    fields = sorted(
        (set().union(*(row.keys() for row in nominal_rows)) & set().union(*(row.keys() for row in anomaly_rows)))
        - {"time_s", "packet_receive_time_s", "packet_sequence", "packet_valid"}
    )
    fields = [field for field in fields if any(field.startswith(prefix) for prefix in FAMILY_PREFIXES[subsystem])]
    pre_noise: dict[str, float] = {field: 0.0 for field in fields}
    for time_s in common_times:
        if time_s >= event:
            break
        left = nominal_by_time[time_s]; right = anomaly_by_time[time_s]
        for field in fields:
            a = base._numeric(left.get(field)); b = base._numeric(right.get(field))
            if a is not None and b is not None:
                pre_noise[field] = max(pre_noise[field], abs(b - a))
    detected_time: float | None = None
    detected_families: dict[str, list[str]] = {}
    post_event_families: dict[str, set[str]] = {}
    for time_s in common_times:
        if time_s < event:
            continue
        families: dict[str, list[str]] = {}
        left = nominal_by_time[time_s]; right = anomaly_by_time[time_s]
        for field in fields:
            a = base._numeric(left.get(field)); b = base._numeric(right.get(field))
            if a is None or b is None:
                continue
            threshold = max(base._resolution_for(field), 3.0 * pre_noise[field])
            if abs(b - a) > threshold:
                families.setdefault(_family(field), []).append(field)
                post_event_families.setdefault(_family(field), set()).add(field)
        # Manifestation is the first credible sensor family response.  The
        # separate causal gate still requires at least two families over the
        # complete post-injection trace.
        if families and detected_time is None:
            detected_time = time_s; detected_families = families
    if detected_time is None:
        phase = "not_observed"
        delay = None
    elif detected_time < activation - 1.0:
        phase = "inactive_immediate"
        delay = detected_time - event
    elif detected_time <= activation + PHASE_HOLD_S:
        phase = "activation_transition"
        delay = detected_time - event
    else:
        phase = "active_settled"
        delay = detected_time - event
    injection_phase = str(metadata["fault_injection_phase"])
    if detected_time is None:
        passed = False
    elif injection_phase == "inactive":
        passed = detected_time >= activation - 1.0
    else:
        # Capacity and thermal-state faults can be physically effective at
        # injection yet remain unobservable until a limit is reached.  Preserve
        # that delay rather than imposing an artificial short detection bound.
        passed = detected_time >= event
    return {
        "fault_manifestation_phase": phase,
        "fault_manifestation_time_s": detected_time,
        "time_to_manifestation_s": delay,
        "manifestation_gate": "PASS" if passed else "FAIL",
        "manifestation_observable_families": detected_families,
        "post_event_observable_families": {
            family: sorted(channels) for family, channels in sorted(post_event_families.items())
        },
        "method": "paired_first_family_divergence_above_quantization_and_pre_event_noise_v1",
    }


```
