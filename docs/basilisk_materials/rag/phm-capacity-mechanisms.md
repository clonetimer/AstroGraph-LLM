# 电池容量故障与延迟生效

角色：SIMULATION_KNOWLEDGE_NOT_CURRENT_OBSERVATION。当前源码核对，非人工批准的诊断真值。

manifest_start_s 门控早于该时刻的事件；容量比例作用于 storageCapacity。事件定义、机制开始生效和传感器首次显现应分别记录。

来源：src/sat_sim/bsk_engine/project_native_modules.py:906-978；SHA256：2fa5e544e18d62cb9b08c214cf0e55cd669fd66bf00559bfa6d73f5ae2ef6597。

```python
    @staticmethod
    def _remaining_ratio(event: BSKEventSpec, default: float = 0.5) -> float:
        if "remaining_capacity_ratio" in event.parameters:
            value = event.parameters.get("remaining_capacity_ratio")
        elif "remaining_efficiency_ratio" in event.parameters:
            value = event.parameters.get("remaining_efficiency_ratio")
        elif "remaining_rejection_ratio" in event.parameters:
            value = event.parameters.get("remaining_rejection_ratio")
        elif "remaining_ratio" in event.parameters:
            value = event.parameters.get("remaining_ratio")
        elif "loss_fraction" in event.parameters:
            value = 1.0 - float(event.parameters.get("loss_fraction", 0.0))
        elif "loss_pct" in event.parameters:
            value = 1.0 - float(event.parameters.get("loss_pct", 0.0)) / 100.0
        else:
            value = default
        try:
            return max(0.0, min(float(value), 1.0))
        except (TypeError, ValueError):
            return float(default)

    def UpdateState(self, current_sim_nanos: int) -> None:  # noqa: N802
        capacity_ratio = 1.0
        efficiency_ratio = 1.0
        rejection_ratio = 1.0
        active_events = _active(self.events, current_sim_nanos)
        now_s = float(current_sim_nanos) * 1.0e-9
        open_circuit = False
        pdu_trip = False
        overload_ratio = 1.0
        storage_ratio = 1.0
        instrument_ratio = 1.0
        downlink_ratio = 1.0
        heater_modes: dict[str, str] = {}
        node_heat_bias_w = 0.0
        nozzle_ratio = 1.0
        fuel_leak_rate = self.nominal_fuel_leak_rate
        for event in active_events:
            manifest_start_s = _event_number(event, ("manifest_start_s",), event.start_s)
            if now_s + 1.0e-12 < manifest_start_s:
                continue
            if event.effect == "eps_battery_capacity_loss":
                capacity_ratio = min(capacity_ratio, self._remaining_ratio(event, 0.5))
            elif event.effect == "eps_battery_open_circuit":
                open_circuit = True
            elif event.effect in {"solar_panel_efficiency_loss", "eps_solar_string_open"}:
                efficiency_ratio = min(efficiency_ratio, self._remaining_ratio(event, 0.5))
            elif event.effect == "eps_pdu_channel_trip":
                pdu_trip = True
            elif event.effect == "eps_load_overload":
                overload_ratio = max(overload_ratio, max(_event_number(event, ("load_multiplier",), 2.0), 1.0))
            elif event.effect == "thermal_radiator_rejection_loss":
                rejection_ratio = min(rejection_ratio, self._remaining_ratio(event, 0.5))
            elif event.effect == "thermal_heater_stuck_off":
                heater_modes[str(event.target).split(".")[-1]] = "off"
            elif event.effect == "thermal_heater_stuck_on":
                heater_modes[str(event.target).split(".")[-1]] = "on"
            elif event.effect == "thermal_node_heat_bias":
                node_heat_bias_w = max(node_heat_bias_w, _event_number(event, ("heat_bias_w", "bias_w"), 15.0))
            elif event.effect == "comm_data_transmitter_rate_loss":
                downlink_ratio = min(downlink_ratio, self._remaining_ratio(event, 0.5))
            elif event.effect == "comm_data_storage_capacity_loss":
                storage_ratio = min(storage_ratio, self._remaining_ratio(event, 0.6))
            elif event.effect == "payload_instrument_rate_loss":
                instrument_ratio = min(instrument_ratio, self._remaining_ratio(event, 0.5))
            elif event.effect == "propulsion_thruster_nozzle_blockage":
                nozzle_ratio = min(nozzle_ratio, self._remaining_ratio(event, 0.25))
            elif event.effect == "propulsion_fuel_leak":
                fuel_leak_rate = max(fuel_leak_rate, _event_number(event, ("leak_rate_kg_s",), 1.0e-4))
        self.battery.storageCapacity = self.nominal_capacity * capacity_ratio
        self.solar_panel.panelEfficiency = self.nominal_efficiency * efficiency_ratio
        if self.gate is not None:
            self.gate.force_safe_mode = bool(open_circuit)
```
